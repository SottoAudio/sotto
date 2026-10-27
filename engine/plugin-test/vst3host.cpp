// sotto_vst3host: a small C library that hosts VST3 plugins as mono inserts, for Sotto Voce's engine
// (called from Python with ctypes in the bench app; linked directly by the C++ engine later).
//
// Threads: svh_process runs on the audio/processing thread. svh_set_param, svh_get_state and
// svh_info may be called from any other thread. Parameter changes are handed to the processing
// thread through a small queue it picks up with try_lock, so svh_process never waits.
#include <map>
#include <memory>
#include <mutex>
#include <sstream>

#include "host_core.h"

#if defined(_WIN32)
#define SVH_API extern "C" __declspec(dllexport)
#else
#define SVH_API extern "C" __attribute__((visibility("default")))
#endif

using namespace sotto;

namespace {

std::mutex gInitMx;
IPtr<HostApplication> gHost;
std::map<std::string, VST3::Hosting::Module::Ptr> gModules;   // loaded once, kept for the session

void init() {
    std::lock_guard<std::mutex> l(gInitMx);
    if (!gHost) {
        gHost = owned(new HostApplication());
        PluginContextFactory::instance().setPluginContext(gHost);
        static std::ostringstream quiet;
        PlugProvider::setErrorStream(&quiet);
    }
}

std::string jsonEsc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

int copyOut(const std::string& s, char* buf, int len) {
    if (buf && len > 0) {
        const size_t n = std::min(s.size(), static_cast<size_t>(len - 1));
        std::memcpy(buf, s.data(), n);
        buf[n] = 0;
    }
    return static_cast<int>(s.size()) + 1;
}

std::vector<char> pack(const State& s) {
    std::vector<char> o(16 + 8 + s.component.size() + 8 + s.controller.size());
    char* p = o.data();
    std::memcpy(p, "SOTTOVST3STATE01", 16); p += 16;
    uint64_t a = s.component.size(), b = s.controller.size();
    std::memcpy(p, &a, 8); p += 8; if (a) std::memcpy(p, s.component.data(), a); p += a;
    std::memcpy(p, &b, 8); p += 8; if (b) std::memcpy(p, s.controller.data(), b);
    return o;
}

bool unpack(const void* data, int len, State& s) {
    const char* p = static_cast<const char*>(data);
    if (len < 32 || std::memcmp(p, "SOTTOVST3STATE01", 16)) return false;
    uint64_t a = 0, b = 0;
    std::memcpy(&a, p + 16, 8);
    if (24 + a + 8 > static_cast<uint64_t>(len)) return false;
    s.component.assign(p + 24, p + 24 + a);
    std::memcpy(&b, p + 24 + a, 8);
    if (24 + a + 8 + b > static_cast<uint64_t>(len)) return false;
    s.controller.assign(p + 32 + a, p + 32 + a + b);
    return true;
}

struct Handle {
    VST3::Hosting::Module::Ptr mod;
    VST3::Hosting::ClassInfo ci;
    Instance inst;
    std::mutex mx;
    std::vector<std::pair<ParamID, double>> pending;
    ParameterChanges changes;
    long blocks = 0, failures = 0;
};

VST3::Hosting::Module::Ptr loadModule(const std::string& path, std::string& err) {
    std::lock_guard<std::mutex> l(gInitMx);
    auto it = gModules.find(path);
    if (it != gModules.end()) return it->second;
    auto m = VST3::Hosting::Module::create(path, err);
    if (m) gModules[path] = m;
    return m;
}

}  // namespace

SVH_API int svh_version() { return 1; }

SVH_API void* svh_open(const char* path, const char* cls, double rate, int block, const void* state, int stateLen,
                       char* err, int errLen) {
    init();
    std::string e;
    auto mod = loadModule(path ? path : "", e);
    if (!mod) { copyOut("could not load: " + e, err, errLen); return nullptr; }
    auto h = std::make_unique<Handle>();
    h->mod = mod;
    bool found = false;
    for (const auto& c : mod->getFactory().classInfos()) {
        if (c.category() != kVstAudioEffectClass) continue;
        if (!cls || !*cls || lower(c.name()) == lower(cls)) { h->ci = c; found = true; break; }
    }
    if (!found) { copyOut("no audio effect in this plugin", err, errLen); return nullptr; }
    State st;
    const bool haveState = state && stateLen > 0 && unpack(state, stateLen, st);
    if (!h->inst.open(mod->getFactory(), h->ci, rate, block, haveState ? &st : nullptr, e)) {
        copyOut(e, err, errLen);
        return nullptr;
    }
    return h.release();
}

SVH_API void svh_close(void* p) { delete static_cast<Handle*>(p); }

SVH_API int svh_process(void* p, const float* in, float* out, int n) {
    auto* h = static_cast<Handle*>(p);
    IParameterChanges* ch = nullptr;
    if (h->mx.try_lock()) {
        if (!h->pending.empty()) {
            h->changes.clearQueue();
            for (const auto& [id, v] : h->pending) {
                int32 qi = 0, pi = 0;
                if (auto* q = h->changes.addParameterData(id, qi)) q->addPoint(0, v, pi);
            }
            h->pending.clear();
            ch = &h->changes;
        }
        h->mx.unlock();
    }
    const bool ok = h->inst.process(in, out, n, ch);
    ++h->blocks;
    if (!ok) ++h->failures;
    return ok ? 1 : 0;
}

SVH_API int svh_latency(void* p) {
    auto* h = static_cast<Handle*>(p);
    return static_cast<int>(h->inst.proc->getLatencySamples());
}

SVH_API int svh_set_param(void* p, unsigned id, double norm) {
    auto* h = static_cast<Handle*>(p);
    norm = std::min(1.0, std::max(0.0, norm));
    if (h->inst.ctrl) h->inst.ctrl->setParamNormalized(id, norm);
    std::lock_guard<std::mutex> l(h->mx);
    h->pending.emplace_back(id, norm);
    return 1;
}

// Name, maker, mono, latency and every visible control with its value as the plugin shows it.
SVH_API int svh_info(void* p, char* buf, int len) {
    auto* h = static_cast<Handle*>(p);
    std::ostringstream o;
    o << "{\"name\":\"" << jsonEsc(h->ci.name()) << "\",\"vendor\":\"" << jsonEsc(h->ci.vendor()) << "\",\"version\":\""
      << jsonEsc(h->ci.version()) << "\",\"type\":\"" << jsonEsc(h->ci.subCategoriesString()) << "\",\"mono_native\":"
      << (h->inst.monoNative ? "true" : "false") << ",\"latency\":" << h->inst.proc->getLatencySamples() << ",\"blocks\":" << h->blocks
      << ",\"failures\":" << h->failures << ",\"params\":[";
    bool first = true;
    for (const auto& q : params(h->inst.ctrl)) {
        if (q.hidden) continue;
        o << (first ? "" : ",") << "{\"id\":" << q.id << ",\"name\":\"" << jsonEsc(q.title) << "\",\"units\":\"" << jsonEsc(q.units)
          << "\",\"value\":\"" << jsonEsc(q.value) << "\",\"norm\":" << q.normalized << ",\"steps\":" << q.steps << "}";
        first = false;
    }
    o << "]}";
    return copyOut(o.str(), buf, len);
}

// The text the plugin shows for a normalized value (for slider labels).
SVH_API int svh_param_text(void* p, unsigned id, double norm, char* buf, int len) {
    auto* h = static_cast<Handle*>(p);
    String128 s {};
    if (!h->inst.ctrl || h->inst.ctrl->getParamStringByValue(id, norm, s) != kResultOk) return copyOut("", buf, len);
    return copyOut(u8(s), buf, len);
}

// Saved settings (component + controller). Call with len 0 to get the size, then again.
SVH_API int svh_get_state(void* p, void* buf, int len) {
    auto* h = static_cast<Handle*>(p);
    State st;
    if (!h->inst.getState(st)) return -1;
    const auto packed = pack(st);
    if (buf && len >= static_cast<int>(packed.size())) std::memcpy(buf, packed.data(), packed.size());
    return static_cast<int>(packed.size());
}
