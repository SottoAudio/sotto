/* sotto_audio_io: the soundcard side of Sotto Voce's engine, in C, the way a DAW does it.
 *
 * Core Audio (through PortAudio) calls sio_cycle() on its real-time thread for every soundcard
 * buffer. It never waits, allocates, locks or calls into Python. It only:
 *   - copies the inputs into a lock-free ring the engine thread reads 10 ms blocks from;
 *   - plays the engine's output from a second ring, a fixed number of samples behind (the
 *     safety margin);
 *   - when the engine hasn't written a sample in time, plays each talker's input straight through
 *     (dry) instead, with a short crossfade both ways, silences every other output, and latches
 *     HEARTBEAT off: it stays off until the app calls sio_clear_heartbeat() after a manual re-arm;
 *   - records each cycle's timing for the diagnostics file.
 *
 * The engine thread (Python) calls sio_read() for each 10 ms block, does its work, and calls
 * sio_write(). Both rings are single-producer, single-consumer, with 64-bit sample counters.
 *
 * PortAudio is loaded at run time from the same library file the Python side already uses
 * (python-sounddevice's), so device numbers match and no PortAudio headers are needed to build.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  include <windows.h>
#  include <timeapi.h>              /* timeBeginPeriod: 1 ms sleeps instead of Windows' default ~15 ms */
#  define SIO_API __declspec(dllexport)
#else
#  include <dlfcn.h>
#  include <time.h>
#  include <unistd.h>
#  define SIO_API __attribute__((visibility("default")))
#endif
#ifdef __APPLE__
#  include <mach/mach_time.h>
#endif

#define SIO_VERSION 1

/* ------------------------------------------------------------------ atomics */
#if defined(_MSC_VER)
#  include <intrin.h>
static int64_t ld64(volatile int64_t *p) { return InterlockedCompareExchange64((volatile LONG64 *)p, 0, 0); }
static void st64(volatile int64_t *p, int64_t v) { InterlockedExchange64((volatile LONG64 *)p, v); }
static int32_t ld32(volatile int32_t *p) { return InterlockedCompareExchange((volatile LONG *)p, 0, 0); }
static void st32(volatile int32_t *p, int32_t v) { InterlockedExchange((volatile LONG *)p, v); }
#else
static int64_t ld64(volatile int64_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static void st64(volatile int64_t *p, int64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static int32_t ld32(volatile int32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static void st32(volatile int32_t *p, int32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
#endif

/* The same clock as Python's time.perf_counter(), in seconds. */
static double now_s(void) {
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

static void nap_us(int us) {
#ifdef _WIN32
    (void)us;
    Sleep(1);
#else
    struct timespec ts = {0, (long)us * 1000L};
    nanosleep(&ts, NULL);
#endif
}

/* ------------------------------------------------------------------ PortAudio (loaded at run time) */
typedef struct { int device; int channelCount; unsigned long sampleFormat; double suggestedLatency; void *hostApiSpecificStreamInfo; } PaStreamParameters;
typedef struct { double inputBufferAdcTime; double currentTime; double outputBufferDacTime; } PaStreamCallbackTimeInfo;
typedef struct { int structVersion; double inputLatency; double outputLatency; double sampleRate; } PaStreamInfo;
typedef int PaStreamCallback(const void *, void *, unsigned long, const PaStreamCallbackTimeInfo *, unsigned long, void *);
typedef int (*fn_init)(void);
typedef int (*fn_open)(void **, const PaStreamParameters *, const PaStreamParameters *, double, unsigned long, unsigned long, PaStreamCallback *, void *);
typedef int (*fn_stream)(void *);
typedef const char *(*fn_err)(int);
typedef const PaStreamInfo *(*fn_info)(void *);
#define paFloat32 ((unsigned long)0x00000001)

/* ------------------------------------------------------------------ the engine's state */
#define TRACE_ROWS 16384          /* ~54 s of 160-sample cycles */
#define TRACE_COLS 6              /* wall time, capture time, frames, callback ms, status bits, margin frames */
#define RAMP 120                  /* crossfade between engine and fallback: 2.5 ms at 48 kHz */

typedef struct {
    int n_in, n_out, block, fs;
    int64_t prefill;              /* the safety margin: how far the output trails the input, in samples */
    int64_t cap;                  /* ring capacity in frames */
    float *in_ring, *out_ring;
    volatile int64_t in_w;        /* written by the callback */
    volatile int64_t in_r;        /* engine only */
    volatile int64_t out_w;       /* published by the engine */
    volatile int64_t out_r;       /* the callback's play position */
    int64_t eng_pos;              /* input position of the block the engine is working on */
    volatile int32_t started;     /* the engine has written its first block */
    volatile int32_t hb_block;    /* HEARTBEAT latched off */
    /* fallback */
    int n_map, *map_in, *map_out, hb_ch;
    float g;                      /* 1 = engine output, 0 = fallback */
    float *held;                  /* last engine sample per output (faded out on a fallback) */
    /* counters */
    volatile int64_t cycles, underrun_cycles, fallback_frames, overruns, dropped_blocks, resyncs;
    /* timing record */
    double trace[TRACE_ROWS][TRACE_COLS];
    volatile int64_t trace_n;
    /* PortAudio */
    void *pa, *stream;
    int pa_inited;                /* our own Pa_Initialize, undone in sio_close (so a later rescan sees new devices) */
    fn_init pa_term;
    fn_stream pa_start, pa_stop, pa_close;
    fn_err pa_err;
    fn_info pa_info;
    char err[512];
} sio_t;

SIO_API int sio_version(void) { return SIO_VERSION; }

SIO_API sio_t *sio_create(int n_in, int n_out, int block, int fs, int prefill, int capacity) {
    if (n_in < 1 || n_out < 1 || block < 1 || prefill < 0) return NULL;
    sio_t *s = (sio_t *)calloc(1, sizeof(sio_t));
    if (!s) return NULL;
    s->n_in = n_in; s->n_out = n_out; s->block = block; s->fs = fs; s->prefill = prefill;
    s->cap = capacity > prefill + 8 * block ? capacity : prefill + 8 * block;
    s->in_ring = (float *)calloc((size_t)(s->cap * n_in), sizeof(float));
    s->out_ring = (float *)calloc((size_t)(s->cap * n_out), sizeof(float));
    s->held = (float *)calloc((size_t)n_out, sizeof(float));
    if (!s->in_ring || !s->out_ring || !s->held) { free(s->in_ring); free(s->out_ring); free(s->held); free(s); return NULL; }
    s->out_w = prefill;           /* the margin starts as silence the engine is "ahead" by */
#ifdef _WIN32
    timeBeginPeriod(1);
#endif
    s->g = 1.0f;
    s->hb_ch = -1;
    return s;
}

/* Dry fallback: talker input map_in[k] plays on output map_out[k]; hb = HEARTBEAT output (-1 = none). */
SIO_API int sio_set_fallback(sio_t *s, int n, const int *map_in, const int *map_out, int hb) {
    int *mi = (int *)malloc(sizeof(int) * (n > 0 ? n : 1)), *mo = (int *)malloc(sizeof(int) * (n > 0 ? n : 1));
    if (!mi || !mo) { free(mi); free(mo); return -1; }
    for (int k = 0; k < n; k++) {
        if (map_in[k] < 0 || map_in[k] >= s->n_in || map_out[k] < 0 || map_out[k] >= s->n_out) { free(mi); free(mo); return -2; }
        mi[k] = map_in[k]; mo[k] = map_out[k];
    }
    free(s->map_in); free(s->map_out);
    s->map_in = mi; s->map_out = mo; s->n_map = n; s->hb_ch = hb;
    return 0;
}

/* ------------------------------------------------------------------ the real-time side */
/* One soundcard buffer: in (frames x n_in) interleaved, out (frames x n_out) interleaved. */
SIO_API void sio_cycle(sio_t *s, const float *in, float *out, int frames, double adc_time, int status) {
    double t0 = now_s();
    const int ni = s->n_in, no = s->n_out;
    /* inputs into the ring (if the engine has fallen a whole ring behind, it resyncs on its side) */
    int64_t w = s->in_w;
    for (int f = 0; f < frames; f++) {
        float *dst = s->in_ring + ((w + f) % s->cap) * ni;
        if (in) memcpy(dst, in + (size_t)f * ni, sizeof(float) * ni);
        else memset(dst, 0, sizeof(float) * ni);
    }
    st64(&s->in_w, w + frames);
    if (w + frames - ld64(&s->in_r) > s->cap) s->overruns++;

    /* outputs from the engine's ring, or the fallback */
    int64_t r = s->out_r, avail = ld64(&s->out_w);
    int started = ld32(&s->started), short_cycle = 0;
    const float step = 1.0f / RAMP;
    for (int f = 0; f < frames; f++) {
        float *o = out + (size_t)f * no;
        int have = (r + f) < avail;
        if (have) {
            const float *src = s->out_ring + ((r + f) % s->cap) * no;
            if (s->g >= 1.0f) memcpy(o, src, sizeof(float) * no);
            for (int c = 0; c < no; c++) s->held[c] = src[c];
            if (s->g < 1.0f) {
                s->g = s->g + step > 1.0f ? 1.0f : s->g + step;
                for (int c = 0; c < no; c++) o[c] = s->g * src[c];
            }
        } else {
            short_cycle = 1;
            s->g = s->g - step < 0.0f ? 0.0f : s->g - step;
            for (int c = 0; c < no; c++) o[c] = s->g * s->held[c];
        }
        if (s->g < 1.0f && s->n_map) {               /* dry talkers, the other way round */
            const float *x = in ? in + (size_t)f * ni : NULL;
            for (int k = 0; k < s->n_map; k++)
                o[s->map_out[k]] += (1.0f - s->g) * (x ? x[s->map_in[k]] : 0.0f);
        }
        if (ld32(&s->hb_block) && s->hb_ch >= 0) o[s->hb_ch] = 0.0f;
    }
    st64(&s->out_r, r + frames);
    if (short_cycle && started) {
        s->underrun_cycles++;
        st32(&s->hb_block, 1);                        /* latched: only a manual re-arm clears it */
        if (s->hb_ch >= 0)
            for (int f = 0; f < frames; f++) out[(size_t)f * no + s->hb_ch] = 0.0f;
    }
    if (s->g < 1.0f) s->fallback_frames += frames;
    s->cycles++;

    int64_t n = s->trace_n;
    double *row = s->trace[n % TRACE_ROWS];
    row[0] = t0; row[1] = adc_time; row[2] = frames; row[3] = 1000.0 * (now_s() - t0);
    row[4] = status; row[5] = (double)(avail - (r + frames));
    st64(&s->trace_n, n + 1);
}

/* ------------------------------------------------------------------ the engine's side */
/* Waits (sleeping, never spinning hard) for the next block of inputs. Returns 1 with a block in
 * buf (frames x n_in), 0 on timeout. A block whose output would already be too late to play is
 * skipped (counted), so the output always stays exactly `prefill` samples behind the input. */
SIO_API int sio_read(sio_t *s, float *buf, int frames, int timeout_ms) {
    double give_up = now_s() + timeout_ms / 1000.0;
    for (;;) {
        int64_t w = ld64(&s->in_w), r = ld64(&s->in_r);
        if (w - r > s->cap - frames) {                /* lost a whole ring: start again from now */
            r = w - frames;
            s->resyncs++;
        }
        if (w - r >= frames) {
            if (r + s->prefill + frames <= ld64(&s->out_r)) {
                st64(&s->in_r, r + frames);           /* too late to be heard: skip it */
                if (s->out_w < r + frames + s->prefill) st64(&s->out_w, r + frames + s->prefill);
                s->dropped_blocks++;
                continue;
            }
            for (int f = 0; f < frames; f++)
                memcpy(buf + (size_t)f * s->n_in, s->in_ring + ((r + f) % s->cap) * s->n_in, sizeof(float) * s->n_in);
            s->eng_pos = r;
            st64(&s->in_r, r + frames);
            return 1;
        }
        if (now_s() >= give_up) return 0;
        nap_us(250);
    }
}

/* The engine's output for the block it last read (frames x n_out). */
SIO_API int sio_write(sio_t *s, const float *buf, int frames) {
    int64_t at = s->eng_pos + s->prefill;
    for (int f = 0; f < frames; f++)
        memcpy(s->out_ring + ((at + f) % s->cap) * s->n_out, buf + (size_t)f * s->n_out, sizeof(float) * s->n_out);
    if (at + frames > ld64(&s->out_w)) st64(&s->out_w, at + frames);
    st32(&s->started, 1);
    return 0;
}

SIO_API void sio_clear_heartbeat(sio_t *s) { st32(&s->hb_block, 0); }
SIO_API int sio_heartbeat_blocked(sio_t *s) { return ld32(&s->hb_block); }

/* Counters: cycles, underrun cycles, fallback frames, input overruns, skipped blocks, resyncs,
 * margin now (frames of engine output waiting), trace rows written. */
SIO_API void sio_stats(sio_t *s, int64_t *v) {
    v[0] = s->cycles; v[1] = s->underrun_cycles; v[2] = s->fallback_frames; v[3] = s->overruns;
    v[4] = s->dropped_blocks; v[5] = s->resyncs; v[6] = ld64(&s->out_w) - ld64(&s->out_r); v[7] = ld64(&s->trace_n);
}

/* Copies trace rows [from, trace_n) (at most max rows, the newest if more) into buf; returns how
 * many, and in *end the row count it copied up to (the next call's `from`). */
SIO_API int sio_trace(sio_t *s, int64_t from, double *buf, int max, int64_t *end) {
    int64_t n = ld64(&s->trace_n);
    if (end) *end = n;
    if (from < n - TRACE_ROWS) from = n - TRACE_ROWS;
    if (from < 0) from = 0;
    if (n - from > max) from = n - max;
    int k = 0;
    for (int64_t i = from; i < n; i++, k++) memcpy(buf + (size_t)k * TRACE_COLS, s->trace[i % TRACE_ROWS], sizeof(double) * TRACE_COLS);
    return k;
}

SIO_API const char *sio_error(sio_t *s) { return s->err; }

/* ------------------------------------------------------------------ the soundcard (PortAudio) */
static int pa_callback(const void *in, void *out, unsigned long frames, const PaStreamCallbackTimeInfo *ti,
                       unsigned long status, void *user) {
    sio_cycle((sio_t *)user, (const float *)in, (float *)out, (int)frames, ti ? ti->inputBufferAdcTime : NAN, (int)status);
    return 0;                                          /* paContinue */
}

static void *sym(void *lib, const char *name) {
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

/* Opens and starts the soundcard: `pa_path` is the PortAudio library file, `device` its device
 * number, `io` frames per buffer, `latency` in seconds (PortAudio's suggested latency). On
 * success also returns the latencies PortAudio reports (seconds). Returns 0, or <0 with sio_error(). */
SIO_API int sio_open(sio_t *s, const char *pa_path, int device, int io, double latency, double *in_lat, double *out_lat) {
#ifdef _WIN32
    s->pa = (void *)LoadLibraryA(pa_path);
#else
    s->pa = dlopen(pa_path, RTLD_NOW | RTLD_GLOBAL);
#endif
    if (!s->pa) { snprintf(s->err, sizeof s->err, "couldn't load PortAudio from %s", pa_path); return -1; }
    fn_init pa_init = (fn_init)sym(s->pa, "Pa_Initialize");
    s->pa_term = (fn_init)sym(s->pa, "Pa_Terminate");
    fn_open pa_open = (fn_open)sym(s->pa, "Pa_OpenStream");
    s->pa_start = (fn_stream)sym(s->pa, "Pa_StartStream");
    s->pa_stop = (fn_stream)sym(s->pa, "Pa_StopStream");
    s->pa_close = (fn_stream)sym(s->pa, "Pa_CloseStream");
    s->pa_err = (fn_err)sym(s->pa, "Pa_GetErrorText");
    s->pa_info = (fn_info)sym(s->pa, "Pa_GetStreamInfo");
    if (!pa_init || !pa_open || !s->pa_start || !s->pa_stop || !s->pa_close || !s->pa_err) {
        snprintf(s->err, sizeof s->err, "PortAudio at %s is missing functions", pa_path); return -2;
    }
    int e = pa_init();
    if (e) { snprintf(s->err, sizeof s->err, "Pa_Initialize: %s", s->pa_err(e)); return -3; }
    s->pa_inited = 1;
    PaStreamParameters ip = {device, s->n_in, paFloat32, latency, NULL};
    PaStreamParameters op = {device, s->n_out, paFloat32, latency, NULL};
    e = pa_open(&s->stream, &ip, &op, (double)s->fs, (unsigned long)io, 0, pa_callback, s);
    if (e) { snprintf(s->err, sizeof s->err, "Pa_OpenStream: %s", s->pa_err(e)); s->stream = NULL; return -4; }
    if (s->pa_info && in_lat && out_lat) {
        const PaStreamInfo *si = s->pa_info(s->stream);
        if (si) { *in_lat = si->inputLatency; *out_lat = si->outputLatency; }
    }
    e = s->pa_start(s->stream);
    if (e) { snprintf(s->err, sizeof s->err, "Pa_StartStream: %s", s->pa_err(e)); s->pa_close(s->stream); s->stream = NULL; return -5; }
    return 0;
}

SIO_API void sio_close(sio_t *s) {
    if (s->stream) { s->pa_stop(s->stream); s->pa_close(s->stream); s->stream = NULL; }
    /* PortAudio reads the list of soundcards when it's first initialised, and only again once
     * every user has terminated it: let go of ours, or a soundcard started later stays invisible. */
    if (s->pa_inited && s->pa_term) { s->pa_term(); s->pa_inited = 0; }
}

SIO_API void sio_destroy(sio_t *s) {
    if (!s) return;
    sio_close(s);
#ifdef _WIN32
    timeEndPeriod(1);
#endif
    free(s->in_ring); free(s->out_ring); free(s->held); free(s->map_in); free(s->map_out);
    free(s);
}
