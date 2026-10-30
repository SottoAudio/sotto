/* sotto engine: Sotto Voce's 10 ms engine in C (stage 3 of the engine-in-C plan).
 *
 * One real-time thread, started and stopped from Python, that never takes Python's lock:
 *   read a block from the soundcard rings (audio_io.c) -> mix in test sources (rendered ahead by
 *   Python) -> swap blocks with the processing workers through their shared memory -> the
 *   fail-safe core (core.c) -> write the block back -> copy the phones and the recording channels
 *   into rings for Python to take away -> one timing row for the diagnostics file.
 *
 * Python talks to it through:
 *   - a command ring (arm, disarm, trip, faults, phones plan, headroom, worker tables), applied at
 *     the start of a block, or within ~20 ms when no audio is arriving; each command's result can
 *     be waited for;
 *   - a few flags it writes directly (the checks Python makes: network, soundcard watchdog; which
 *     workers are ready);
 *   - rings it reads: events (from the core), timing rows, recording, phones;
 *   - a ring it writes: test-source blocks.
 * Every ring is single-producer / single-consumer; Python serialises its own producers.
 *
 * Processing hand-over: the same slots and semaphores as workers.Pool.exchange, byte for byte:
 * block k goes into the worker's input slot, its semaphore is posted, and the result for block
 * k - headroom is taken if the worker has written it; otherwise that worker's talkers get the dry
 * input from the same delay. A talker switching between the two is faded over 2.5 ms. Three late
 * blocks in a row make processing unhealthy, which disarms (latched), as in Python.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  include <windows.h>
#  define SE_API __declspec(dllexport)
#else
#  include <pthread.h>
#  include <semaphore.h>
#  include <time.h>
#  include <dlfcn.h>
#  include <sys/resource.h>
#  define SE_API __attribute__((visibility("default")))
#endif
#ifdef __APPLE__
#  include <mach/mach.h>
#  include <mach/mach_time.h>
#  include <mach/thread_policy.h>
#endif

/* ------------------------------------------------------------------ the other two parts */
typedef struct sio sio_t;
typedef struct SC SC;
int sio_read(sio_t *s, float *buf, int frames, int timeout_ms);
int sio_write(sio_t *s, const float *buf, int frames);
void sio_clear_heartbeat(sio_t *s);
int sio_heartbeat_blocked(sio_t *s);
void sio_stats(sio_t *s, int64_t *v);
double sc_time(SC *c);
int sc_request_arm(SC *c, int talker);
void sc_request_disarm(SC *c);
void sc_disarm_why(SC *c, int why, int arg);
void sc_trip_talker(SC *c, int talker);
void sc_inject(SC *c, int fault, double seconds);
void sc_clear_faults(SC *c);
void sc_set_solo(SC *c, int nl, const int *lk, const int *lc, int nr, const int *rk, const int *rc,
                 double level_db, int dim, double dim_db, double limit_db);
int sc_process(SC *c, const float *in, float *out, int frames, int status, const double *returns,
               const double *dry, int ext_ok);
int sc_armed(SC *c);
void sc_not_green(SC *c);

/* ------------------------------------------------------------------ atomics and clocks */
#if defined(_MSC_VER)
#  include <intrin.h>
static int64_t ld64(volatile int64_t *p) { return InterlockedCompareExchange64((volatile LONG64 *)p, 0, 0); }
static void st64(volatile int64_t *p, int64_t v) { InterlockedExchange64((volatile LONG64 *)p, v); }
static int32_t ld32(volatile int32_t *p) { return InterlockedCompareExchange((volatile LONG *)p, 0, 0); }
static void st32(volatile int32_t *p, int32_t v) { InterlockedExchange((volatile LONG *)p, v); }
static void fence(void) { MemoryBarrier(); }
#else
static int64_t ld64(volatile int64_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static void st64(volatile int64_t *p, int64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static int32_t ld32(volatile int32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static void st32(volatile int32_t *p, int32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static void fence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
#endif

static double wall_s(void) {          /* the same clock as Python's time.perf_counter() */
#if defined(__APPLE__)
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
#elif defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

static double thread_cpu_s(void) {
#ifdef _WIN32
    FILETIME a, b, k, u;
    if (!GetThreadTimes(GetCurrentThread(), &a, &b, &k, &u)) return NAN;
    return ((double)(((uint64_t)k.dwHighDateTime << 32) | k.dwLowDateTime) +
            (double)(((uint64_t)u.dwHighDateTime << 32) | u.dwLowDateTime)) * 1e-7;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts)) return NAN;
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

static double process_cpu_s(void) {
#ifdef _WIN32
    FILETIME a, b, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &a, &b, &k, &u)) return NAN;
    return ((double)(((uint64_t)k.dwHighDateTime << 32) | k.dwLowDateTime) +
            (double)(((uint64_t)u.dwHighDateTime << 32) | u.dwLowDateTime)) * 1e-7;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts)) return NAN;
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

/* ------------------------------------------------------------------ semaphores (multiprocessing's) */
static void sem_give(void *h) {
#ifdef _WIN32
    ReleaseSemaphore((HANDLE)h, 1, NULL);
#else
    sem_post((sem_t *)h);
#endif
}

static void sem_take_nowait(void *h) {           /* a barrier, as Python's acquire(False) */
#ifdef _WIN32
    WaitForSingleObject((HANDLE)h, 0);
#else
    sem_trywait((sem_t *)h);
#endif
}

/* ------------------------------------------------------------------ the engine */
#define HOP 480
#define RAMP 120
#define SLOTS 4
#define HEADER 16
#define MAX_HEADROOM 3
#define NHIST (MAX_HEADROOM + 2)
#define MAXW 16
#define MAXT 64
#define NCMD 256
#define TIMING_COLS 10
#define TIMING_ROWS 8192
#define NREC 512                   /* 5 s of recording: rides out a stalled main thread */
#define NPHONES 128
#define NTEST 32
#define STAGES 5

enum { CMD_ARM = 1, CMD_DISARM, CMD_TRIP, CMD_INJECT, CMD_CLEAR, CMD_SOLO, CMD_HEADROOM, CMD_WORKER, CMD_FREEZE,
       CMD_TESTS };
/* A test-source loop as Python hands it over (its samples stay alive in Python until the next plan
 * has been taken): the input channel it replaces, the samples, how many, the gain. */
typedef struct { int32_t ch; int32_t pad; const float *x; int64_t n; double gain; } TLoop;
#define MAXLOOPS 64
/* an arm request's answer: 1 armed; 0 not green long enough (or everyone not armed yet); -1 no such
 * talker; -2 too old (asked more than ARM_MAX_AGE ago); -3 a check outside the core is red (the
 * network, the soundcard, processing, the engine just late); -4 no audio arriving */
#define ARM_MAX_AGE 0.2
enum { WHY_LATE = 5, WHY_PROC = 6 };

typedef struct { int32_t op, a, b; double x; void *p0, *p1, *p2; } Cmd;
typedef struct {
    int row0, g;
    uint8_t *base;                 /* the worker's shared memory */
    void *in_sem, *out_sem;
} Worker;

typedef struct SE {
    sio_t *io; SC *core;
    int n_in, n_out, block, fs, nt;
    int tin[MAXT], pgm;
    /* thread */
    volatile int32_t run, running;
    double period_ms, comp_ms, cons_ms;
    void *volatile wg_new;         /* an os_workgroup_t from Python (owned from here on) */
    volatile int32_t wg_gen;
    int32_t wg_seen;
    double last_block;             /* wall clock of the last block run */
    char wg_note[160], rt_note[160];
#ifdef _WIN32
    HANDLE th;
#else
    pthread_t th;
#endif
    /* commands */
    Cmd cmd[NCMD];
    volatile int64_t cmd_w, cmd_r;
    double result[NCMD];
    int st_nl, st_nr, st_lk[128], st_lc[128], st_rk[128], st_rc[128], st_dim;
    double st_level, st_dim_db, st_limit;
    Worker st_worker;
    int st_wid;
    /* flags Python writes */
    volatile int32_t ext_ok, ready[MAXW];
    /* processing */
    int proc_on, nw, headroom;
    Worker w[MAXW];
    float *hist, *out_buf, *dry_buf, *last_out, fade[RAMP];
    uint8_t *missed, *was_missed;
    double *ret, *dry;             /* frames x nt, for the core */
    int64_t pblocks, grace_until, misses, miss_run, max_miss_run, good_run;
    int32_t proc_ok, live, last_missed;
    /* late engine */
    int64_t late_seen, late_total;
    double late_until;
    int was_armed;
    /* buffers */
    float *inb, *outb;
    /* rings out */
    double timing[TIMING_ROWS][TIMING_COLS];
    volatile int64_t tim_w, tim_r;
    int nrec, rec_src[MAXT * 2 + 8], rec_ch[MAXT * 2 + 8];   /* src 0 = input, 1 = output */
    float *rec; volatile int64_t rec_w, rec_r; int64_t rec_dropped;
    int mon_l, mon_r;
    float *ph; volatile int64_t ph_w, ph_r;
    /* test sources, played here: the loops and the clock every loop follows */
    TLoop loops[MAXLOOPS];
    int nloops;
    int64_t tclock;
    uint8_t tch[512];                  /* which inputs the plan replaces */
    /* ring in: test sources (older path, unused) */
    float *test; uint8_t *test_mask; volatile int64_t test_w, test_r; int64_t test_short;
    int64_t blocks;
} SE;

static void apply_cmd(SE *e, Cmd *c, double *res);
static void workgroup_symbols(void);
static void run_block(SE *e);

SE_API SE *se_create(sio_t *io, SC *core, int n_in, int n_out, int block, int fs, int nt, const int *tin, int pgm,
                     int nrec, const int *rec_src, const int *rec_ch, int mon_l, int mon_r) {
    if (nt > MAXT || nrec > MAXT * 2 + 8 || block <= 0) return NULL;
    SE *e = calloc(1, sizeof(SE));
    if (!e) return NULL;
    e->io = io; e->core = core; e->n_in = n_in; e->n_out = n_out; e->block = block; e->fs = fs; e->nt = nt; e->pgm = pgm;
    for (int i = 0; i < nt; i++) e->tin[i] = tin[i];
    e->nrec = nrec;
    for (int i = 0; i < nrec; i++) { e->rec_src[i] = rec_src[i]; e->rec_ch[i] = rec_ch[i]; }
    e->mon_l = mon_l; e->mon_r = mon_r;
    e->ext_ok = 1;
    e->period_ms = 1000.0 * block / fs; e->comp_ms = 4.0; e->cons_ms = e->period_ms;
    int n = nt > 0 ? nt : 1;
    e->inb = calloc((size_t)block * n_in, sizeof(float));
    e->outb = calloc((size_t)block * n_out, sizeof(float));
    e->hist = calloc((size_t)NHIST * n * HOP, sizeof(float));
    e->out_buf = calloc((size_t)n * HOP, sizeof(float));
    e->dry_buf = calloc((size_t)n * HOP, sizeof(float));
    e->last_out = calloc(n, sizeof(float));
    e->missed = calloc(n, 1); e->was_missed = calloc(n, 1);
    e->ret = calloc((size_t)block * n, sizeof(double));
    e->dry = calloc((size_t)block * n, sizeof(double));
    e->rec = calloc((size_t)NREC * block * (nrec > 0 ? nrec : 1), sizeof(float));
    e->ph = calloc((size_t)NPHONES * block * 2, sizeof(float));
    e->test = calloc((size_t)NTEST * block * n_in, sizeof(float));
    e->test_mask = calloc((size_t)NTEST * n_in, 1);
    if (!e->inb || !e->outb || !e->hist || !e->out_buf || !e->dry_buf || !e->last_out || !e->missed ||
        !e->was_missed || !e->ret || !e->dry || !e->rec || !e->ph || !e->test || !e->test_mask) {
        free(e); return NULL;     /* small leak on a failed start-up; never happens in practice */
    }
    for (int i = 0; i < RAMP; i++) e->fade[i] = (float)(1.0 - (double)i / RAMP);   /* linspace(1, 0, RAMP, endpoint=False) */
    e->headroom = 1;
    e->proc_ok = 1;
    e->late_until = -1.0;
    strcpy(e->wg_note, "not joined");
    strcpy(e->rt_note, "not started");
    workgroup_symbols();
    return e;
}

SE_API void se_destroy(SE *e) {
    if (!e) return;
    free(e->inb); free(e->outb); free(e->hist); free(e->out_buf); free(e->dry_buf); free(e->last_out);
    free(e->missed); free(e->was_missed); free(e->ret); free(e->dry); free(e->rec); free(e->ph); free(e->test);
    free(e->test_mask); free(e);
}

/* Processing: nw workers in talker order, the headroom, and whether processing is healthy now. */
SE_API void se_set_proc(SE *e, int on, int nw, int headroom, int ok) {
    e->proc_on = on && e->block == HOP && nw > 0 && nw <= MAXW;
    e->nw = nw;
    e->headroom = headroom < 1 ? 1 : headroom > MAX_HEADROOM ? MAX_HEADROOM : headroom;
    e->proc_ok = ok;
}

/* Before start: a worker's table directly. After start: through se_command(CMD_WORKER). */
SE_API void se_stage_worker(SE *e, int wid, int row0, int g, void *base, void *in_sem, void *out_sem) {
    Worker w = {row0, g, (uint8_t *)base, in_sem, out_sem};
    e->st_worker = w;
    e->st_wid = wid;
    if (!ld32(&e->running) && wid >= 0 && wid < MAXW) e->w[wid] = w;
}

SE_API void se_stage_solo(SE *e, int nl, const int *lk, const int *lc, int nr, const int *rk, const int *rc,
                          double level_db, int dim, double dim_db, double limit_db) {
    if (nl > 128) nl = 128;
    if (nr > 128) nr = 128;
    for (int i = 0; i < nl; i++) { e->st_lk[i] = lk[i]; e->st_lc[i] = lc[i]; }
    for (int i = 0; i < nr; i++) { e->st_rk[i] = rk[i]; e->st_rc[i] = rc[i]; }
    e->st_nl = nl; e->st_nr = nr; e->st_level = level_db; e->st_dim = dim; e->st_dim_db = dim_db; e->st_limit = limit_db;
}

SE_API void se_set_ext(SE *e, int ok) { st32(&e->ext_ok, ok ? 1 : 0); }
SE_API void se_set_ready(SE *e, int wid, int ready) { if (wid >= 0 && wid < MAXW) st32(&e->ready[wid], ready ? 1 : 0); }
/* The soundcard's I/O workgroup (an os_workgroup_t Python read from Core Audio; this side owns it
 * from here on), joined by the engine thread at its next turn. NULL leaves the current one. */
SE_API void se_set_workgroup(SE *e, void *wg) {
    e->wg_new = wg;
    st32(&e->wg_gen, ld32(&e->wg_gen) + 1);
}
SE_API void se_set_timing(SE *e, double comp_ms) { e->comp_ms = comp_ms; }

/* Returns the command's number (wait on it with se_result), or -1 when the ring is full. */
SE_API int64_t se_command_p(SE *e, int op, int a, int b, double x, void *p0, void *p1, void *p2) {
    int64_t w = e->cmd_w;
    if (w - ld64(&e->cmd_r) >= NCMD) return -1;
    Cmd *c = &e->cmd[w % NCMD];
    c->op = op; c->a = a; c->b = b; c->x = x; c->p0 = p0; c->p1 = p1; c->p2 = p2;
    st64(&e->cmd_w, w + 1);
    return w;
}

SE_API int64_t se_command(SE *e, int op, int a, int b, double x) { return se_command_p(e, op, a, b, x, NULL, NULL, NULL); }

SE_API double se_clock(void) { return wall_s(); }

/* 1 when command n has been applied (its result in *res), else 0. */
SE_API int se_result(SE *e, int64_t n, double *res) {
    if (ld64(&e->cmd_r) <= n) return 0;
    *res = e->result[n % NCMD];
    return 1;
}

/* Every check outside the core, as one: the ones Python makes (network, soundcard), processing,
 * the engine having been late, and audio arriving at all. */
static int outside_ok(SE *e) {
    if (!ld32(&e->ext_ok)) return -3;
    double t = sc_time(e->core);
    if (e->late_until >= 0 && t < e->late_until) return -3;
    if (e->proc_on && !e->proc_ok) return -3;
    if (wall_s() - e->last_block > 0.05) return -4;
    return 1;
}

static void drain_commands(SE *e) {
    if (outside_ok(e) != 1) sc_not_green(e->core);   /* a red check restarts the 5 s, as in Python */
    int64_t r = e->cmd_r, w = ld64(&e->cmd_w);
    while (r < w) {
        double res = 0;
        apply_cmd(e, &e->cmd[r % NCMD], &res);
        e->result[r % NCMD] = res;
        r++;
        st64(&e->cmd_r, r);
    }
}

static void apply_cmd(SE *e, Cmd *c, double *res) {
    switch (c->op) {
    case CMD_ARM: {
        if (wall_s() - c->x > ARM_MAX_AGE) { *res = -2; break; }     /* nobody is waiting for this answer */
        int why = outside_ok(e);
        if (why != 1) { sc_not_green(e->core); *res = why; break; }
        int ok = sc_request_arm(e->core, c->a);
        if (ok == 1 && sio_heartbeat_blocked(e->io)) sio_clear_heartbeat(e->io);   /* a manual re-arm */
        *res = ok;
        break;
    }
    case CMD_DISARM: sc_disarm_why(e->core, c->a, c->b); break;
    case CMD_TRIP: sc_trip_talker(e->core, c->a); break;
    case CMD_INJECT: sc_inject(e->core, c->a, c->x); break;
    case CMD_CLEAR: sc_clear_faults(e->core); break;
    case CMD_SOLO:
        sc_set_solo(e->core, e->st_nl, e->st_lk, e->st_lc, e->st_nr, e->st_rk, e->st_rc,
                    e->st_level, e->st_dim, e->st_dim_db, e->st_limit);
        break;
    case CMD_HEADROOM: {
        int h = c->a < 1 ? 1 : c->a > MAX_HEADROOM ? MAX_HEADROOM : c->a;
        for (int i = 0; i < e->nw; i++) if (e->w[i].base) ((int64_t *)e->w[i].base)[1] = h;
        e->grace_until = e->pblocks + 2 * MAX_HEADROOM;
        e->headroom = h;
        break;
    }
    case CMD_WORKER:                    /* by value: wid, row0 | g << 16, shared memory, semaphores */
        if (c->a >= 0 && c->a < MAXW) {
            Worker w = {c->b & 0xFFFF, c->b >> 16, (uint8_t *)c->p0, c->p1, c->p2};
            e->w[c->a] = w;
        }
        break;
    case CMD_TESTS: {                   /* a new plan: p0 = TLoop[a] (copied here), or a = 0 for none */
        int n = c->a < 0 ? 0 : c->a > MAXLOOPS ? MAXLOOPS : c->a;
        const TLoop *src = (const TLoop *)c->p0;
        memset(e->tch, 0, sizeof e->tch);
        for (int i = 0; i < n; i++) {
            e->loops[i] = src[i];
            if (src[i].ch >= 0 && src[i].ch < e->n_in && src[i].ch < 512) e->tch[src[i].ch] = 1;
        }
        e->nloops = n;
        break;
    }
    case CMD_FREEZE: {                  /* the console's "freeze" test: deliberately miss deadlines */
        int ms = c->a < 0 ? 0 : c->a > 2000 ? 2000 : c->a;
#ifdef _WIN32
        Sleep(ms);
#else
        struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
        nanosleep(&ts, NULL);
#endif
        break;
    }
    }
}

/* ------------------------------------------------------------------ the processing hand-over */
static void exchange(SE *e, const float *in, int frames) {
    int n = e->nt, h = e->headroom;
    int64_t k = e->pblocks++;
    float *mics = e->hist + (size_t)(k % NHIST) * n * HOP;           /* history[k % nh] = mics */
    for (int i = 0; i < n; i++)
        for (int f = 0; f < HOP; f++) mics[(size_t)i * HOP + f] = in[(size_t)f * e->n_in + e->tin[i]];
    int any_missed = 0, n_missed = 0;
    memset(e->missed, 0, (size_t)(n > 0 ? n : 0));
    for (int wi = 0; wi < e->nw; wi++) {
        Worker *w = &e->w[wi];
        if (!w->base) continue;
        size_t in_slot = 16 + (size_t)4 * HOP * (w->g + 1);
        size_t out_slot = 16 + (size_t)4 * HOP * 2 * w->g;
        uint8_t *is = w->base + HEADER + (size_t)(k % SLOTS) * in_slot;
        memcpy(is + 16, mics + (size_t)w->row0 * HOP, sizeof(float) * HOP * w->g);
        float *ref = (float *)(is + 16 + (size_t)4 * HOP * w->g);
        if (e->pgm >= 0) for (int f = 0; f < HOP; f++) ref[f] = in[(size_t)f * e->n_in + e->pgm];
        else memset(ref, 0, sizeof(float) * HOP);
        fence();
        ((volatile int64_t *)is)[0] = k;
        ((volatile int64_t *)w->base)[0] = k;
        sem_give(w->in_sem);
        int64_t j = k - h;
        uint8_t *os = w->base + HEADER + (size_t)SLOTS * in_slot + (size_t)(((j % SLOTS) + SLOTS) % SLOTS) * out_slot;
        int got = 0;
        if (j >= 0 && ((volatile int64_t *)os)[0] == j) {
            sem_take_nowait(w->out_sem);
            fence();
            memcpy(e->out_buf + (size_t)w->row0 * HOP, os + 16, sizeof(float) * HOP * w->g);
            memcpy(e->dry_buf + (size_t)w->row0 * HOP, os + 16 + (size_t)4 * HOP * w->g, sizeof(float) * HOP * w->g);
            fence();
            got = ((volatile int64_t *)os)[0] == j;
        }
        if (!got) {
            any_missed = 1;
            n_missed++;
            memset(e->missed + w->row0, 1, w->g);
            for (int r = w->row0; r < w->row0 + w->g; r++) {
                if (k - h - 1 >= 0) {
                    const float *old = e->hist + (size_t)((k - h - 1) % NHIST) * n * HOP + (size_t)r * HOP;
                    memcpy(e->out_buf + (size_t)r * HOP, old, sizeof(float) * HOP);
                    memcpy(e->dry_buf + (size_t)r * HOP, old, sizeof(float) * HOP);
                } else {
                    memset(e->out_buf + (size_t)r * HOP, 0, sizeof(float) * HOP);
                    memset(e->dry_buf + (size_t)r * HOP, 0, sizeof(float) * HOP);
                }
            }
        }
    }
    e->last_missed = n_missed;
    /* declick */
    for (int r = 0; r < n; r++) {
        float *o = e->out_buf + (size_t)r * HOP;
        if (e->missed[r] != e->was_missed[r]) {
            float step = e->last_out[r] - o[0];
            for (int f = 0; f < RAMP && f < HOP; f++) o[f] += step * e->fade[f];
        }
        e->was_missed[r] = e->missed[r];
        e->last_out[r] = o[HOP - 1];
    }
    int all_ready = 1;
    for (int wi = 0; wi < e->nw; wi++) if (!ld32(&e->ready[wi])) all_ready = 0;
    if (!any_missed) {
        e->miss_run = 0;
        e->good_run++;
        if (!e->proc_ok && e->good_run >= 100 && all_ready) e->proc_ok = 1;
    } else {
        e->good_run = 0;
        if (e->live && e->pblocks > 50 && k >= e->grace_until) {
            e->misses++;
            e->miss_run++;
            if (e->miss_run > e->max_miss_run) e->max_miss_run = e->miss_run;
            if (e->miss_run >= 3 && e->proc_ok) e->proc_ok = 0;
        }
    }
    if (!e->live && all_ready) e->live = 1;
    /* the core takes frames x talkers */
    for (int f = 0; f < frames; f++)
        for (int i = 0; i < n; i++) {
            e->ret[(size_t)f * n + i] = e->out_buf[(size_t)i * HOP + f];
            e->dry[(size_t)f * n + i] = e->dry_buf[(size_t)i * HOP + f];
        }
}

/* ------------------------------------------------------------------ one block */
static void run_block(SE *e) {
    int B = e->block, ni = e->n_in, no = e->n_out;
    double t0 = wall_s(), c0 = thread_cpu_s(), p0 = process_cpu_s(), st[STAGES] = {0};
    double ta = t0;
    e->last_block = t0;
    drain_commands(e);
    int armed = sc_armed(e->core);
    /* test sources, as TestSources.apply: each planned input is replaced by the sum of its loops,
     * all following one clock. Never while armed. */
    if (e->nloops && !armed) {
        for (int ch = 0; ch < ni && ch < 512; ch++)
            if (e->tch[ch]) for (int f = 0; f < B; f++) e->inb[(size_t)f * ni + ch] = 0.0f;
        for (int i = 0; i < e->nloops; i++) {
            const TLoop *lp = &e->loops[i];
            if (lp->n <= 0 || lp->ch < 0 || lp->ch >= ni) continue;
            float g = (float)lp->gain;
            int64_t s = e->tclock % lp->n;
            for (int f = 0; f < B; f++) {
                e->inb[(size_t)f * ni + lp->ch] += lp->x[s] * g;
                if (++s == lp->n) s = 0;
            }
        }
    }
    e->tclock += B;
    double tb_ = wall_s(); st[0] = 1000 * (tb_ - ta); ta = tb_;
    int ext = ld32(&e->ext_ok);
    double t = sc_time(e->core);
    if (e->late_until >= 0 && t < e->late_until) ext = 0;
    const double *ret = NULL, *dry = NULL;
    if (e->proc_on) {
        exchange(e, e->inb, B);
        ret = e->ret; dry = e->dry;
        if (!e->proc_ok) {
            ext = 0;
            if (armed) sc_disarm_why(e->core, WHY_PROC, 0);   /* says why, before the core's own check */
        }
    }
    tb_ = wall_s(); st[1] = 1000 * (tb_ - ta); ta = tb_;
    sc_process(e->core, e->inb, e->outb, B, 0, ret, dry, ext);
    tb_ = wall_s(); st[2] = 1000 * (tb_ - ta); ta = tb_;
    if (e->mon_l >= 0) {
        int64_t w = e->ph_w;
        if (w - ld64(&e->ph_r) < NPHONES) {
            float *p = e->ph + (size_t)(w % NPHONES) * B * 2;
            for (int f = 0; f < B; f++) {
                p[2 * f] = e->outb[(size_t)f * no + e->mon_l];
                p[2 * f + 1] = e->outb[(size_t)f * no + (e->mon_r >= 0 ? e->mon_r : e->mon_l)];
            }
            st64(&e->ph_w, w + 1);
        }
    }
    tb_ = wall_s(); st[3] = 1000 * (tb_ - ta); ta = tb_;
    if (e->nrec > 0) {
        int64_t w = e->rec_w;
        if (w - ld64(&e->rec_r) < NREC) {
            float *p = e->rec + (size_t)(w % NREC) * B * e->nrec;
            for (int f = 0; f < B; f++)
                for (int c = 0; c < e->nrec; c++)
                    p[(size_t)f * e->nrec + c] = e->rec_src[c] ? e->outb[(size_t)f * no + e->rec_ch[c]]
                                                               : e->inb[(size_t)f * ni + e->rec_ch[c]];
            st64(&e->rec_w, w + 1);
        } else e->rec_dropped++;
    }
    sio_write(e->io, e->outb, B);
    tb_ = wall_s(); st[4] = 1000 * (tb_ - ta);
    /* the soundcard side played dry because a block was late: a fault (disarm, 5 s not green) */
    int64_t sv[8];
    sio_stats(e->io, sv);
    if (sv[1] > e->late_seen) {
        int64_t n = sv[1] - e->late_seen;
        e->late_total += n;
        e->late_seen = sv[1];
        e->late_until = sc_time(e->core) + 5.0;
        sc_disarm_why(e->core, WHY_LATE, (int)n);
    }
    e->blocks++;
    double t1 = wall_s();
    int64_t w = e->tim_w;
    if (w - ld64(&e->tim_r) < TIMING_ROWS) {
        double *row = e->timing[w % TIMING_ROWS];
        row[0] = t0; row[1] = 1000 * (t1 - t0);
        for (int i = 0; i < STAGES; i++) row[2 + i] = st[i];
        row[7] = e->proc_on ? e->last_missed : 0;
        row[8] = 1000 * (thread_cpu_s() - c0);
        row[9] = 1000 * (process_cpu_s() - p0);
        st64(&e->tim_w, w + 1);
    }
}

/* ------------------------------------------------------------------ the thread */
#ifdef __APPLE__
static void make_realtime(SE *e) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double k = 1e6 * tb.denom / tb.numer;
    thread_time_constraint_policy_data_t p;
    p.period = (uint32_t)(e->period_ms * k);
    p.computation = (uint32_t)(e->comp_ms * k);
    p.constraint = (uint32_t)(e->cons_ms * k);
    p.preemptible = 1;
    kern_return_t rc = thread_policy_set(mach_thread_self(), THREAD_TIME_CONSTRAINT_POLICY, (thread_policy_t)&p,
                                         THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    if (rc == KERN_SUCCESS) snprintf(e->rt_note, sizeof e->rt_note, "real-time (%g ms every %g ms), in C", e->comp_ms, e->period_ms);
    else snprintf(e->rt_note, sizeof e->rt_note, "real-time refused (error %d), in C", rc);
}

#define FOURCC(a, b, c, d) (((uint32_t)(a) << 24) | ((uint32_t)(b) << 16) | ((uint32_t)(c) << 8) | (uint32_t)(d))
typedef int (*fn_wg_join)(void *, void *);
typedef void (*fn_wg_leave)(void *, void *);
typedef void (*fn_release)(void *);
static void *wg_obj; static char wg_token[64];

static fn_wg_join wg_join; static fn_wg_leave wg_leave; static fn_release wg_release;

static void workgroup_symbols(void) {                 /* once, at create time, off the real-time thread */
    wg_join = (fn_wg_join)dlsym(RTLD_DEFAULT, "os_workgroup_join");
    wg_leave = (fn_wg_leave)dlsym(RTLD_DEFAULT, "os_workgroup_leave");
    wg_release = (fn_release)dlsym(RTLD_DEFAULT, "os_release");
}

static void workgroup(SE *e, void *wg) {
    fn_wg_join join = wg_join; fn_wg_leave leave = wg_leave; fn_release release = wg_release;
    if (wg_obj && leave) { leave(wg_obj, wg_token); if (release) release(wg_obj); wg_obj = NULL; }
    if (!wg || !join) { snprintf(e->wg_note, sizeof e->wg_note, wg ? "audio workgroups not available" : "no audio workgroup"); return; }
    memset(wg_token, 0, sizeof wg_token);
    int rc = join(wg, wg_token);
    if (rc) { if (release) release(wg); snprintf(e->wg_note, sizeof e->wg_note, "couldn't join the soundcard's audio workgroup (error %d)", rc); return; }
    wg_obj = wg;
    snprintf(e->wg_note, sizeof e->wg_note, "in the soundcard's audio workgroup");
}
#else
static void make_realtime(SE *e) {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    snprintf(e->rt_note, sizeof e->rt_note, "time-critical priority, in C");
#else
    snprintf(e->rt_note, sizeof e->rt_note, "normal priority, in C");
#endif
}
static void workgroup_symbols(void) {}
static void workgroup(SE *e, void *wg) { snprintf(e->wg_note, sizeof e->wg_note, "not on this system"); }
#endif

static void engine_main(SE *e) {
    make_realtime(e);
    st32(&e->running, 1);
    while (ld32(&e->run)) {
        int32_t gen = ld32(&e->wg_gen);
        if (gen != e->wg_seen) { e->wg_seen = gen; workgroup(e, e->wg_new); }
        if (sio_read(e->io, e->inb, e->block, 20) == 1) run_block(e);
        else drain_commands(e);              /* no audio arriving: still answer Python */
    }
#ifdef __APPLE__
    workgroup(e, NULL);
#endif
    st32(&e->running, 0);
}

#ifdef _WIN32
static DWORD WINAPI thread_fn(LPVOID p) { engine_main((SE *)p); return 0; }
#else
static void *thread_fn(void *p) { engine_main((SE *)p); return NULL; }
#endif

SE_API int se_start(SE *e) {
    if (ld32(&e->running) || ld32(&e->run)) return 0;
    st32(&e->run, 1);
#ifdef _WIN32
    e->th = CreateThread(NULL, 0, thread_fn, e, 0, NULL);
    return e->th ? 0 : -1;
#else
    return pthread_create(&e->th, NULL, thread_fn, e) == 0 ? 0 : -1;
#endif
}

SE_API void se_stop(SE *e) {
    if (!ld32(&e->run)) return;
    st32(&e->run, 0);
#ifdef _WIN32
    WaitForSingleObject(e->th, INFINITE);             /* never free what the thread may still use */
    CloseHandle(e->th);
#else
    pthread_join(e->th, NULL);
#endif
}

/* ------------------------------------------------------------------ reading, from Python */
/* v: running, blocks, proc blocks, misses, miss run, max miss run, proc ok, last missed, late total,
 *    late until (s), rec dropped, test short, headroom, live */
SE_API void se_stats(SE *e, double *v) {
    v[0] = ld32(&e->running); v[1] = (double)e->blocks; v[2] = (double)e->pblocks; v[3] = (double)e->misses;
    v[4] = (double)e->miss_run; v[5] = (double)e->max_miss_run; v[6] = e->proc_ok; v[7] = e->last_missed;
    v[8] = (double)e->late_total; v[9] = e->late_until; v[10] = (double)e->rec_dropped; v[11] = (double)e->test_short;
    v[12] = e->headroom; v[13] = e->live;
}

SE_API const char *se_thread_note(SE *e) { return e->rt_note; }
SE_API const char *se_workgroup_note(SE *e) { return e->wg_note; }

SE_API int se_timing(SE *e, double *rows, int max) {
    int64_t r = e->tim_r, w = ld64(&e->tim_w);
    int n = 0;
    while (r < w && n < max) { memcpy(rows + (size_t)n * TIMING_COLS, e->timing[r % TIMING_ROWS], sizeof(double) * TIMING_COLS); r++; n++; }
    st64(&e->tim_r, r);
    return n;
}

SE_API int se_record(SE *e, float *buf, int max_blocks) {
    int64_t r = e->rec_r, w = ld64(&e->rec_w);
    int n = 0, sz = e->block * e->nrec;
    while (r < w && n < max_blocks) { memcpy(buf + (size_t)n * sz, e->rec + (size_t)(r % NREC) * sz, sizeof(float) * sz); r++; n++; }
    st64(&e->rec_r, r);
    return n;
}

SE_API int se_phones(SE *e, float *buf, int max_blocks) {
    int64_t r = e->ph_r, w = ld64(&e->ph_w);
    int n = 0, sz = e->block * 2;
    while (r < w && n < max_blocks) { memcpy(buf + (size_t)n * sz, e->ph + (size_t)(r % NPHONES) * sz, sizeof(float) * sz); r++; n++; }
    st64(&e->ph_r, r);
    return n;
}

/* Test sources: how many blocks are queued, and one more (frames x n_in, mask per input). */
SE_API int se_test_queued(SE *e) { return (int)(ld64(&e->test_w) - ld64(&e->test_r)); }

SE_API int se_test_push(SE *e, const float *block, const uint8_t *mask) {
    int64_t w = e->test_w;
    if (w - ld64(&e->test_r) >= NTEST) return 0;
    memcpy(e->test + (size_t)(w % NTEST) * e->block * e->n_in, block, sizeof(float) * e->block * e->n_in);
    memcpy(e->test_mask + (size_t)(w % NTEST) * e->n_in, mask, e->n_in);
    st64(&e->test_w, w + 1);
    return 1;
}

/* For the tests: one processing hand-over on a given block (frames x n_in), without the thread or
 * the soundcard; the results as the core would get them (frames x talkers). */
SE_API void se_exchange_test(SE *e, const float *in, double *ret, double *dry) {
    exchange(e, in, e->block);
    memcpy(ret, e->ret, sizeof(double) * e->block * (e->nt > 0 ? e->nt : 1));
    memcpy(dry, e->dry, sizeof(double) * e->block * (e->nt > 0 ? e->nt : 1));
}
