/* sotto core: Sotto Voce's fail-safe core in C (stage 2 of the engine-in-C plan).
 *
 * The same rules and the same arithmetic as failsafe/sotto_failsafe/core.py (FailsafeCore) and
 * arming.py (ArmState), block by block:
 *   - talker inputs pass to their returns (or the processed / delay-matched dry signal), each
 *     return carrying its quiet MARKER tone;
 *   - PROBE is always sent; every PROBE input is checked with the same Goertzel test; a probe
 *     missing longer than the timeout is a fault, which disarms (latched);
 *   - HEARTBEAT only while armed and every check is green;
 *   - re-arming only on request, after arm_hold seconds of green;
 *   - the phones (solo) bus: the taps Python planned, level, dim, limiter.
 *
 * Stage 2 runs it in shadow: Python's core drives the outputs, this one computes the same block
 * from the same input and the app compares them. Nothing here allocates, locks or waits once
 * created, so stage 3 can call it from the C engine thread.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define SC_API __declspec(dllexport)
#else
#  define SC_API __attribute__((visibility("default")))
#endif

#define SC_MAXT 64
#define SC_MAXP 8
#define SC_MAXCH 256
#define SC_MAXTAP 128
#define SC_MAXFRAMES 4096
#define SC_NEVER (-1.0)          /* green_since / last_seen not set (Python's None) */
#define SC_EVENTS 256
#define TWO_PI 6.283185307179586

enum { EV_ARMED = 1, EV_DISARMED, EV_ARM_REFUSED, EV_TALKER_ARMED, EV_TALKER_TRIPPED, EV_FAULT_ENDED };
enum { WHY_PROBE = 1, WHY_STARTING, WHY_EXTERNAL, WHY_REQUEST };

typedef struct { double freq, amp, phase; int fs; } Tone;
typedef struct { double freq, timeout, min_dbfs, dominance, last_seen; } Watch;
typedef struct { double t; int32_t kind, talker, why, arg; } Event;

typedef struct {
    int fs, n_in, n_out, nt, np, nmon;
    int tin[SC_MAXT], tout[SC_MAXT], pin[SC_MAXP], pout, hbout, listen_in, pgm_in, mon[2], marker_on;
    double grace, hold;
    Tone hb, probe, marker[SC_MAXT];
    Watch watch[SC_MAXP];
    int64_t samples, blocks, xruns;
    /* arming */
    int armed, tarmed[SC_MAXT];
    double green_since;
    int why, why_arg;            /* why it isn't green: WHY_* and the probe number */
    /* injected faults: the audio time each ends; <0 = off; INFINITY = until cleared */
    double f_silence, f_mute;
    /* phones bus */
    int nl, nr, lk[SC_MAXTAP], lc[SC_MAXTAP], rk[SC_MAXTAP], rc[SC_MAXTAP];
    double level_db, dim_db, limit_db, solo_gain, limit;
    int dim;
    /* levels */
    double lv_in[SC_MAXCH], lv_out[SC_MAXCH], talker_db[SC_MAXT], listen_db;
    /* events for the app, single producer (this core) / single consumer */
    Event ev[SC_EVENTS];
    volatile int64_t ev_w, ev_r;
    double *x, *sig, *mk, *g, *ls, *rs;   /* scratch, one block */
} SC;

static double dbfs(double v) { return 20.0 * log10(v > 1e-12 ? v : 1e-12); }

static void tone_init(Tone *t, double f, double level, int fs) {
    t->freq = f; t->amp = pow(10.0, level / 20.0); t->phase = 0.0; t->fs = fs;
}

/* Python: amp * sin(phase + w * arange(n)); phase = (phase + w * n) % 2pi */
static void tone_block(Tone *t, double *out, int n) {
    double w = TWO_PI * t->freq / t->fs;
    for (int k = 0; k < n; k++) out[k] = t->amp * sin(t->phase + w * k);
    t->phase = fmod(t->phase + w * n, TWO_PI);
}

static void tone_skip(Tone *t, int n) {
    double w = TWO_PI * t->freq / t->fs;
    t->phase = fmod(t->phase + w * n, TWO_PI);
}

/* signals.tone_present */
static int tone_present(const double *x, int n, int stride, double freq, int fs, double min_dbfs, double dom) {
    if (n <= 0) return 0;
    double w = TWO_PI * freq / fs, re = 0.0, im = 0.0, total = 0.0;
    for (int k = 0; k < n; k++) {
        double v = x[k * stride];
        re += v * cos(w * k);
        im -= v * sin(w * k);
        total += v * v;
    }
    double p = (re * re + im * im) / ((double)n * n);
    double rms = sqrt(2 * p);
    if (dbfs(rms) < min_dbfs) return 0;
    total = total / n + 1e-20;
    double rest = total - rms * rms;
    if (rest < 1e-20) rest = 1e-20;
    return 10 * log10(rms * rms / rest) >= dom;
}

static double missing_s(const Watch *w, double t) {
    if (w->last_seen < 0) return INFINITY;
    double m = t - w->last_seen;
    return m > 0 ? m : 0.0;
}

static void event(SC *c, double t, int kind, int talker, int why, int arg) {
    int64_t w = c->ev_w;
    if (w - c->ev_r >= SC_EVENTS) return;            /* the app stopped reading: drop, never block */
    Event *e = &c->ev[w % SC_EVENTS];
    e->t = t; e->kind = kind; e->talker = talker; e->why = why; e->arg = arg;
#if defined(_MSC_VER)
    c->ev_w = w + 1;
#else
    __atomic_store_n(&c->ev_w, w + 1, __ATOMIC_RELEASE);
#endif
}

static void disarm(SC *c, double t, int why, int arg) {
    int was = c->armed;
    c->armed = 0;
    for (int i = 0; i < c->nt; i++) c->tarmed[i] = 0;
    if (was) event(c, t, EV_DISARMED, -1, why, arg);
}

static double green_for(const SC *c, double t) { return c->green_since < 0 ? 0.0 : t - c->green_since; }

/* ------------------------------------------------------------------ set-up */
SC_API SC *sc_create(int fs, int n_in, int n_out, int nt, const int *tin, const int *tout,
                     int np, const int *pin, int pout, int hbout, int listen_in, int pgm_in,
                     int nmon, const int *mon, int marker_on,
                     double hb_hz, double hb_db, double probe_hz, double probe_db,
                     double marker_hz, double marker_db, double probe_timeout, double hold, double grace) {
    if (nt < 0 || nt > SC_MAXT || np < 0 || np > SC_MAXP || n_in > SC_MAXCH || n_out > SC_MAXCH || nmon > 2) return NULL;
    SC *c = calloc(1, sizeof(SC));
    if (!c) return NULL;
    c->fs = fs; c->n_in = n_in; c->n_out = n_out; c->nt = nt; c->np = np; c->nmon = nmon;
    for (int i = 0; i < nt; i++) { c->tin[i] = tin[i]; c->tout[i] = tout[i]; }
    for (int i = 0; i < np; i++) c->pin[i] = pin[i];
    for (int i = 0; i < nmon; i++) c->mon[i] = mon[i];
    c->pout = pout; c->hbout = hbout; c->listen_in = listen_in; c->pgm_in = pgm_in; c->marker_on = marker_on;
    c->grace = grace; c->hold = hold;
    tone_init(&c->hb, hb_hz, hb_db, fs);
    tone_init(&c->probe, probe_hz, probe_db, fs);
    for (int i = 0; i < nt; i++) tone_init(&c->marker[i], marker_hz, marker_db, fs);
    for (int i = 0; i < np; i++) {
        c->watch[i].freq = probe_hz; c->watch[i].timeout = probe_timeout;
        c->watch[i].min_dbfs = -50.0; c->watch[i].dominance = 12.0; c->watch[i].last_seen = SC_NEVER;
    }
    c->green_since = SC_NEVER;
    c->why = WHY_STARTING;
    c->f_silence = c->f_mute = -1.0;
    c->level_db = 0.0; c->dim_db = -20.0; c->limit_db = -3.0; c->limit = 1.0;
    for (int i = 0; i < SC_MAXCH; i++) c->lv_in[i] = c->lv_out[i] = -180.0;
    for (int i = 0; i < SC_MAXT; i++) c->talker_db[i] = -120.0;
    c->listen_db = -120.0;
    size_t n = SC_MAXFRAMES;
    c->x = malloc(sizeof(double) * n * (n_in > 0 ? n_in : 1));
    c->sig = malloc(sizeof(double) * n); c->mk = malloc(sizeof(double) * n); c->g = malloc(sizeof(double) * n);
    c->ls = malloc(sizeof(double) * n); c->rs = malloc(sizeof(double) * n);
    if (!c->x || !c->sig || !c->mk || !c->g || !c->ls || !c->rs) {
        free(c->x); free(c->sig); free(c->mk); free(c->g); free(c->ls); free(c->rs); free(c);
        return NULL;
    }
    return c;
}

SC_API void sc_destroy(SC *c) {
    if (!c) return;
    free(c->x); free(c->sig); free(c->mk); free(c->g); free(c->ls); free(c->rs); free(c);
}

/* ------------------------------------------------------------------ control (one thread at a time) */
SC_API double sc_time(SC *c) { return (double)c->samples / c->fs; }

/* talker < 0 = everyone. Returns 1 accepted, 0 refused (checks not green long enough, or a talker
 * asked before everyone is armed), -1 no such talker. */
SC_API int sc_request_arm(SC *c, int talker) {
    double t = sc_time(c);
    if (green_for(c, t) < c->hold) { event(c, t, EV_ARM_REFUSED, talker, 0, 0); return 0; }
    if (talker < 0) {
        c->armed = 1;
        for (int i = 0; i < c->nt; i++) c->tarmed[i] = 1;
        event(c, t, EV_ARMED, -1, 0, 0);
        return 1;
    }
    if (talker >= c->nt) return -1;
    if (!c->armed) { event(c, t, EV_ARM_REFUSED, talker, 0, 0); return 0; }
    c->tarmed[talker] = 1;
    event(c, t, EV_TALKER_ARMED, talker, 0, 0);
    return 1;
}

SC_API void sc_request_disarm(SC *c) { disarm(c, sc_time(c), WHY_REQUEST, 0); }

/* A disarm that says why: 5 = the engine was late, 6 = processing can't keep up, 3 = a check
 * Sotto Voce makes outside the core (its reason stays on the Python side). */
SC_API void sc_disarm_why(SC *c, int why, int arg) { disarm(c, sc_time(c), why ? why : WHY_REQUEST, arg); }

SC_API int sc_armed(SC *c) { return c->armed; }

SC_API void sc_trip_talker(SC *c, int talker) {
    if (talker < 0 || talker >= c->nt || !c->tarmed[talker]) return;
    c->tarmed[talker] = 0;
    event(c, sc_time(c), EV_TALKER_TRIPPED, talker, 0, 0);
}

/* fault: 1 silence_all, 2 mute_returns. seconds < 0 = until cleared. */
SC_API void sc_inject(SC *c, int fault, double seconds) {
    double end = seconds < 0 ? INFINITY : sc_time(c) + seconds;
    if (fault == 1) c->f_silence = end;
    else if (fault == 2) c->f_mute = end;
}

SC_API void sc_clear_faults(SC *c) { c->f_silence = c->f_mute = -1.0; }

/* The taps Python planned (core.set_solo): kind 0 = input, 1 = output, 0-based channels. */
SC_API void sc_set_solo(SC *c, int nl, const int *lk, const int *lc, int nr, const int *rk, const int *rc,
                        double level_db, int dim, double dim_db, double limit_db) {
    if (nl > SC_MAXTAP) nl = SC_MAXTAP;
    if (nr > SC_MAXTAP) nr = SC_MAXTAP;
    for (int i = 0; i < nl; i++) { c->lk[i] = lk[i]; c->lc[i] = lc[i]; }
    for (int i = 0; i < nr; i++) { c->rk[i] = rk[i]; c->rc[i] = rc[i]; }
    c->nl = nl; c->nr = nr;
    c->level_db = level_db; c->dim = dim; c->dim_db = dim_db; c->limit_db = limit_db;
}

/* ------------------------------------------------------------------ one block
 * in: frames x n_in, out: frames x n_out (float32, interleaved as numpy C order).
 * returns / dry: frames x nt (float64) or NULL (then each talker passes straight through).
 * ext_ok: the checks Sotto Voce makes outside this core (processing, network, engine late). */
SC_API int sc_process(SC *c, const float *in, float *out, int frames, int status,
                      const double *returns, const double *dry, int ext_ok) {
    if (frames <= 0 || frames > SC_MAXFRAMES) return -1;
    double t = sc_time(c);
    int ni = c->n_in, no = c->n_out;
    if (status) c->xruns++;
    if (c->f_silence >= 0 && t >= c->f_silence) { c->f_silence = -1.0; event(c, t, EV_FAULT_ENDED, -1, 1, 0); }
    if (c->f_mute >= 0 && t >= c->f_mute) { c->f_mute = -1.0; event(c, t, EV_FAULT_ENDED, -1, 2, 0); }
    double *x = c->x;
    for (int i = 0; i < frames * ni; i++) x[i] = in[i];

    /* checks */
    for (int p = 0; p < c->np; p++)
        if (tone_present(x + c->pin[p], frames, ni, c->watch[p].freq, c->fs, c->watch[p].min_dbfs, c->watch[p].dominance))
            c->watch[p].last_seen = t;
    int healthy = 1, why = 0, arg = 0;
    if (t >= c->grace) {
        for (int p = 0; p < c->np; p++)
            if (!(missing_s(&c->watch[p], t) <= c->watch[p].timeout)) { healthy = 0; why = WHY_PROBE; arg = p; break; }
    } else { healthy = 0; why = WHY_STARTING; }
    if (healthy && !ext_ok) { healthy = 0; why = WHY_EXTERNAL; }
    c->why = why; c->why_arg = arg;
    if (healthy) { if (c->green_since < 0) c->green_since = t; }
    else { c->green_since = SC_NEVER; if (c->armed) disarm(c, t, why, arg); }

    /* outputs */
    memset(out, 0, sizeof(float) * frames * no);
    int silent = c->f_silence >= 0, mute = c->f_mute >= 0;
    for (int i = 0; i < c->nt; i++) {
        int ci = c->tin[i], co = c->tout[i];
        double ss = 0.0;
        for (int k = 0; k < frames; k++) { double v = x[k * ni + ci]; ss += v * v; }
        c->talker_db[i] = dbfs(sqrt(ss / frames) + 1e-12);
        double *s = c->sig;
        if (!returns) for (int k = 0; k < frames; k++) s[k] = x[k * ni + ci];
        else if (c->armed && !c->tarmed[i]) for (int k = 0; k < frames; k++) s[k] = dry[k * c->nt + i];
        else for (int k = 0; k < frames; k++) s[k] = returns[k * c->nt + i];
        tone_block(&c->marker[i], c->mk, frames);
        if (c->marker_on) for (int k = 0; k < frames; k++) s[k] += c->mk[k];
        if (!(silent || mute)) for (int k = 0; k < frames; k++) out[k * no + co] = (float)s[k];
    }
    if (!silent) { tone_block(&c->probe, c->sig, frames); for (int k = 0; k < frames; k++) out[k * no + c->pout] = (float)c->sig[k]; }
    else tone_skip(&c->probe, frames);
    tone_block(&c->hb, c->sig, frames);
    if (c->armed && !silent) for (int k = 0; k < frames; k++) out[k * no + c->hbout] = (float)c->sig[k];
    if (c->listen_in >= 0) {
        double ss = 0.0;
        for (int k = 0; k < frames; k++) { double v = x[k * ni + c->listen_in]; ss += v * v; }
        c->listen_db = dbfs(sqrt(ss / frames) + 1e-12);
    }
    if (c->nmon > 0) {
        int any = c->nl || c->nr;
        double target = (silent || !any) ? 0.0 : pow(10.0, (c->level_db + (c->dim ? c->dim_db : 0.0)) / 20.0);
        double a = c->solo_gain, step = (target - a) / frames, add = (target - c->solo_gain) / frames;
        for (int k = 0; k < frames; k++) c->g[k] = (k * step + a) + add;
        c->solo_gain = target;
        for (int side = 0; side < 2; side++) {
            int n = side ? c->nr : c->nl;
            const int *kk = side ? c->rk : c->lk, *cc = side ? c->rc : c->lc;
            double *y = side ? c->rs : c->ls;
            for (int k = 0; k < frames; k++) y[k] = 0.0;
            for (int j = 0; j < n; j++) {
                int ch = cc[j];
                if (kk[j] == 0 && ch < ni) for (int k = 0; k < frames; k++) y[k] += x[k * ni + ch];
                else if (kk[j] == 1 && ch < no) for (int k = 0; k < frames; k++) y[k] += (double)out[k * no + ch];
            }
            for (int k = 0; k < frames; k++) y[k] *= c->g[k];
        }
        double ceiling = pow(10.0, c->limit_db / 20.0), peak = 0.0;
        for (int k = 0; k < frames; k++) {
            double l = fabs(c->ls[k]), r = fabs(c->rs[k]);
            if (l > peak) peak = l;
            if (r > peak) peak = r;
        }
        double need = peak <= ceiling ? 1.0 : ceiling / peak;
        double rel = c->limit * pow(10.0, 6.0 / 20.0 * frames / c->fs);
        if (rel > 1.0) rel = 1.0;
        c->limit = need < rel ? need : rel;
        for (int side = 0; side < c->nmon; side++) {
            double *y = side ? c->rs : c->ls;
            int ch = c->mon[side];
            for (int k = 0; k < frames; k++) {
                double v = y[k] * c->limit;
                out[k * no + ch] = (float)(v < -ceiling ? -ceiling : v > ceiling ? ceiling : v);
            }
        }
    }
    for (int ch = 0; ch < ni && ch < SC_MAXCH; ch++) {
        double ss = 0.0;
        for (int k = 0; k < frames; k++) { double v = x[k * ni + ch]; ss += v * v; }
        c->lv_in[ch] = 20.0 * log10(sqrt(ss / frames) + 1e-9);
    }
    for (int ch = 0; ch < no && ch < SC_MAXCH; ch++) {
        double ss = 0.0;
        for (int k = 0; k < frames; k++) { double v = out[k * no + ch]; ss += v * v; }
        c->lv_out[ch] = 20.0 * log10(sqrt(ss / frames) + 1e-9);
    }
    c->samples += frames;
    c->blocks++;
    return healthy;
}

/* ------------------------------------------------------------------ reading the state */
/* v: armed, heartbeat on, green (0/1), green_for (s), why, why arg, xruns, blocks */
SC_API void sc_state(SC *c, double *v, int *tarmed) {
    double t = sc_time(c);
    v[0] = c->armed;
    v[1] = c->armed && c->f_silence < 0;
    v[2] = c->green_since >= 0;
    v[3] = green_for(c, t);
    v[4] = c->why;
    v[5] = c->why_arg;
    v[6] = (double)c->xruns;
    v[7] = (double)c->blocks;
    for (int i = 0; i < c->nt; i++) tarmed[i] = c->tarmed[i];
}

SC_API double sc_probe_missing(SC *c, int p) { return p >= 0 && p < c->np ? missing_s(&c->watch[p], sc_time(c)) : INFINITY; }

SC_API void sc_levels(SC *c, double *lv_in, double *lv_out, double *talker_db) {
    memcpy(lv_in, c->lv_in, sizeof(double) * c->n_in);
    memcpy(lv_out, c->lv_out, sizeof(double) * c->n_out);
    memcpy(talker_db, c->talker_db, sizeof(double) * c->nt);
}

/* Events since the last call: up to max rows of (t, kind, talker, why, arg). */
SC_API int sc_events(SC *c, double *rows, int max) {
#if defined(_MSC_VER)
    int64_t w = c->ev_w;
#else
    int64_t w = __atomic_load_n(&c->ev_w, __ATOMIC_ACQUIRE);
#endif
    int n = 0;
    while (c->ev_r < w && n < max) {
        Event *e = &c->ev[c->ev_r % SC_EVENTS];
        rows[n * 5] = e->t; rows[n * 5 + 1] = e->kind; rows[n * 5 + 2] = e->talker;
        rows[n * 5 + 3] = e->why; rows[n * 5 + 4] = e->arg;
        n++;
        c->ev_r++;
    }
    return n;
}
