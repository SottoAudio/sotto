// sotto-plugin-test: load a VST3 plugin the way Sotto's engine will and check it is usable live.
//
//   sotto-plugin-test --list
//   sotto-plugin-test "/Library/Audio/Plug-Ins/VST3/CEDAR Voxis.vst3" --in caller.wav --out caller_voxis.wav
//
// For each plugin it reports: whether it runs as a mono insert, its latency, its controls, whether
// its settings can be saved and restored (the operator-profile feature), and how much CPU 1..N
// instances take at the block size Sotto uses. Nothing here is Sotto-specific DSP: it is the
// hosting layer, tested on its own before it goes into the engine.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivstcomponent.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include "pluginterfaces/vst/vstspeaker.h"
#include "public.sdk/source/common/memorystream.h"
#include "public.sdk/source/vst/hosting/hostclasses.h"
#include "public.sdk/source/vst/hosting/module.h"
#include "public.sdk/source/vst/hosting/parameterchanges.h"
#include "public.sdk/source/vst/hosting/plugprovider.h"
#include "public.sdk/source/vst/hosting/processdata.h"
#include "public.sdk/source/vst/utility/stringconvert.h"
#include "wav.h"

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace {

const char* kUsage = R"(sotto-plugin-test: check a VST3 plugin works as a Sotto insert

Usage:
  sotto-plugin-test --list                      list the VST3 plugins on this computer
  sotto-plugin-test PLUGIN.vst3 [options]       test one plugin

Options:
  --class NAME          which plugin inside the bundle (default: the first audio effect)
  --in FILE.wav         audio to process (default: 10 s of test signal). One channel is used.
  --channel N           channel of --in to use, from 1 (default 1)
  --out FILE.wav        write the processed audio (32-bit float, mono)
  --compensate          remove the plugin's latency from the start of --out
  --set NAME=VALUE      set a control before processing, e.g. --set "Attenuation=12 dB" or
                        --set Delay=0.25 (0..1). Repeatable.
  --rate HZ             sample rate when there is no --in (default 48000)
  --block N             samples per block (default 480 = 10 ms at 48 kHz)
  --instances N         how many instances to time together (default 4)
  --save-state FILE     save the plugin's settings after processing
  --load-state FILE     start from saved settings
  --json                print the results as JSON
  --help
)";

std::string u8(const TChar* s) {
    return VST3::StringConvert::convert(std::u16string(reinterpret_cast<const char16_t*>(s)));
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string jsonEsc(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

struct State {
    std::vector<char> component, controller;
    bool empty() const { return component.empty(); }
};

bool saveStateFile(const std::string& path, const State& s) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint64_t a = s.component.size(), b = s.controller.size();
    f.write("SOTTOVST3STATE01", 16);
    f.write(reinterpret_cast<const char*>(&a), 8); f.write(s.component.data(), static_cast<std::streamsize>(a));
    f.write(reinterpret_cast<const char*>(&b), 8); f.write(s.controller.data(), static_cast<std::streamsize>(b));
    return static_cast<bool>(f);
}

bool loadStateFile(const std::string& path, State& s) {
    std::ifstream f(path, std::ios::binary);
    char magic[16];
    if (!f.read(magic, 16) || std::memcmp(magic, "SOTTOVST3STATE01", 16)) return false;
    uint64_t a = 0, b = 0;
    f.read(reinterpret_cast<char*>(&a), 8); s.component.resize(a); f.read(s.component.data(), static_cast<std::streamsize>(a));
    f.read(reinterpret_cast<char*>(&b), 8); s.controller.resize(b); f.read(s.controller.data(), static_cast<std::streamsize>(b));
    return static_cast<bool>(f);
}

IPtr<MemoryStream> streamOf(const std::vector<char>& bytes) {
    auto s = owned(new MemoryStream());
    int32 written = 0;
    if (!bytes.empty()) s->write(const_cast<char*>(bytes.data()), static_cast<int32>(bytes.size()), &written);
    s->seek(0, IBStream::kIBSeekSet, nullptr);
    return s;
}

std::vector<char> bytesOf(MemoryStream& s) {
    return std::vector<char>(s.getData(), s.getData() + s.getSize());
}

// One running copy of the plugin: a mono insert, the way the engine will use it.
struct Instance {
    IPtr<PlugProvider> provider;
    IPtr<IComponent> comp;
    IPtr<IEditController> ctrl;
    IPtr<IAudioProcessor> proc;
    HostProcessData data;
    ProcessContext ctx {};
    int32 inCh = 0, outCh = 0, latency = 0, tail = 0;
    bool monoNative = false, active = false;

    bool open(const VST3::Hosting::PluginFactory& factory, const VST3::Hosting::ClassInfo& ci, double rate,
              int32 block, const State* restore, std::string& err) {
        provider = owned(new PlugProvider(factory, ci, true));
        if (!provider->initialize()) { err = "the plugin did not start (initialise failed)"; return false; }
        comp = provider->getComponentPtr();
        ctrl = provider->getControllerPtr();
        proc = FUnknownPtr<IAudioProcessor>(comp.get()).getInterface();
        if (!comp || !proc) { err = "not an audio processor"; return false; }
        const int32 nIn = comp->getBusCount(kAudio, kInput), nOut = comp->getBusCount(kAudio, kOutput);
        if (nIn < 1 || nOut < 1) { err = "has no audio input and output, so it can't be an insert"; return false; }
        std::vector<SpeakerArrangement> ins(nIn, SpeakerArr::kEmpty), outs(nOut, SpeakerArr::kEmpty);
        for (int32 i = 0; i < nIn; ++i) proc->getBusArrangement(kInput, i, ins[i]);
        for (int32 i = 0; i < nOut; ++i) proc->getBusArrangement(kOutput, i, outs[i]);
        ins[0] = outs[0] = SpeakerArr::kMono;
        monoNative = proc->setBusArrangements(ins.data(), nIn, outs.data(), nOut) == kResultTrue;
        if (!monoNative) {
            ins[0] = outs[0] = SpeakerArr::kStereo;
            proc->setBusArrangements(ins.data(), nIn, outs.data(), nOut);
        }
        BusInfo bi {};
        comp->getBusInfo(kAudio, kInput, 0, bi); inCh = bi.channelCount;
        comp->getBusInfo(kAudio, kOutput, 0, bi); outCh = bi.channelCount;
        if (inCh < 1 || outCh < 1) { err = "main bus has no channels"; return false; }
        for (int32 i = 0; i < nIn; ++i) comp->activateBus(kAudio, kInput, i, i == 0);
        for (int32 i = 0; i < nOut; ++i) comp->activateBus(kAudio, kOutput, i, i == 0);
        if (restore && !restore->empty() && !setState(*restore)) { err = "the plugin refused the saved settings"; return false; }
        if (proc->canProcessSampleSize(kSample32) != kResultTrue) { err = "can't process 32-bit float audio"; return false; }
        ProcessSetup setup {kRealtime, kSample32, block, rate};
        if (proc->setupProcessing(setup) != kResultOk) { err = "refused the sample rate or block size"; return false; }
        if (comp->setActive(true) != kResultOk) { err = "could not be switched on (setActive)"; return false; }
        active = true;
        latency = static_cast<int32>(proc->getLatencySamples());
        tail = static_cast<int32>(proc->getTailSamples());
        data.prepare(*comp, block, kSample32);
        data.processMode = kRealtime;
        data.processContext = &ctx;
        ctx.sampleRate = rate;
        ctx.tempo = 120.0;
        ctx.state = ProcessContext::kPlaying | ProcessContext::kTempoValid | ProcessContext::kContTimeValid;
        proc->setProcessing(true);
        return true;
    }

    bool process(const float* in, float* out, int32 n, IParameterChanges* changes) {
        data.numSamples = n;
        for (int32 c = 0; c < inCh; ++c) std::memcpy(data.inputs[0].channelBuffers32[c], in, sizeof(float) * n);
        for (int32 b = 1; b < data.numInputs; ++b)
            for (int32 c = 0; c < data.inputs[b].numChannels; ++c)
                std::memset(data.inputs[b].channelBuffers32[c], 0, sizeof(float) * n);
        data.inputParameterChanges = changes;
        const tresult r = proc->process(data);
        ctx.projectTimeSamples += n;
        ctx.continousTimeSamples += n;
        std::memcpy(out, data.outputs[0].channelBuffers32[0], sizeof(float) * n);
        return r == kResultOk;
    }

    bool getState(State& s) {
        auto a = owned(new MemoryStream());
        if (comp->getState(a) != kResultOk) return false;
        s.component = bytesOf(*a);
        s.controller.clear();
        if (ctrl) {
            auto b = owned(new MemoryStream());
            if (ctrl->getState(b) == kResultOk) s.controller = bytesOf(*b);
        }
        return true;
    }

    bool setState(const State& s) {
        auto a = streamOf(s.component);
        if (comp->setState(a) != kResultOk) return false;
        if (ctrl) {
            a->seek(0, IBStream::kIBSeekSet, nullptr);
            ctrl->setComponentState(a);
            if (!s.controller.empty()) { auto b = streamOf(s.controller); ctrl->setState(b); }
        }
        return true;
    }

    void close() {
        if (proc && active) { proc->setProcessing(false); comp->setActive(false); }
        active = false;
        data.unprepare();
        proc = nullptr; ctrl = nullptr; comp = nullptr; provider = nullptr;
    }
    ~Instance() { close(); }
};

struct Param {
    ParamID id;
    std::string title, units, value;
    double normalized;
    int32 steps;
    bool hidden;
};

std::vector<Param> params(IEditController* ctrl) {
    std::vector<Param> out;
    if (!ctrl) return out;
    for (int32 i = 0; i < ctrl->getParameterCount(); ++i) {
        ParameterInfo pi {};
        if (ctrl->getParameterInfo(i, pi) != kResultOk) continue;
        String128 s {};
        const double v = ctrl->getParamNormalized(pi.id);
        ctrl->getParamStringByValue(pi.id, v, s);
        out.push_back({pi.id, u8(pi.title), u8(pi.units), u8(s), v, pi.stepCount,
                       (pi.flags & ParameterInfo::kIsHidden) != 0 || (pi.flags & ParameterInfo::kIsReadOnly) != 0});
    }
    return out;
}

// --set "Name=12 dB" -> (id, normalized). Accepts the plugin's own text, or a plain 0..1 value.
bool resolveSet(IEditController* ctrl, const std::string& name, const std::string& value, ParamID& id, double& norm,
                std::string& err) {
    for (const auto& p : params(ctrl)) {
        if (lower(p.title) != lower(name) && std::to_string(p.id) != name) continue;
        id = p.id;
        String128 s {};
        VST3::StringConvert::convert(value, s);
        ParamValue v = 0;
        if (ctrl->getParamValueByString(p.id, s, v) == kResultOk) { norm = v; return true; }
        char* end = nullptr;
        const double d = std::strtod(value.c_str(), &end);
        if (end && *end == 0 && d >= 0.0 && d <= 1.0) { norm = d; return true; }
        err = "control \"" + name + "\" didn't accept \"" + value + "\" (try a 0..1 value)";
        return false;
    }
    err = "no control called \"" + name + "\" (run without --set to see the list)";
    return false;
}

std::vector<float> testSignal(double rate, double seconds) {
    // Speech-like bursts (band-limited noise, 1.5 s on / 0.5 s off) over a steady -45 dBFS bed.
    std::mt19937 rng(7);
    std::normal_distribution<float> g(0.f, 1.f);
    const size_t n = static_cast<size_t>(rate * seconds);
    std::vector<float> x(n);
    float lp = 0.f, bp = 0.f;
    for (size_t i = 0; i < n; ++i) {
        const float w = g(rng);
        lp += 0.15f * (w - lp);
        bp = lp - bp * 0.2f;
        const double t = i / rate;
        const bool talking = std::fmod(t, 2.0) < 1.5;
        x[i] = 0.0056f * g(rng) + (talking ? 0.25f * bp * static_cast<float>(0.6 + 0.4 * std::sin(2 * 3.14159265358979323846 * 4 * t)) : 0.f);
    }
    return x;
}

double peakDb(const std::vector<float>& a) {
    float m = 0.f;
    for (float v : a) m = std::max(m, std::fabs(v));
    return m > 0 ? 20.0 * std::log10(m) : -200.0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string plugin, cls, inPath, outPath, saveState, loadState;
    std::vector<std::pair<std::string, std::string>> sets;
    double rate = 48000.0;
    int32 block = 480, nInst = 4, channel = 1;
    bool list = false, json = false, compensate = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::cerr << what << " needs a value\n"; std::exit(1); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { std::cout << kUsage; return 0; }
        else if (a == "--list") list = true;
        else if (a == "--json") json = true;
        else if (a == "--compensate") compensate = true;
        else if (a == "--class") cls = next("--class");
        else if (a == "--in") inPath = next("--in");
        else if (a == "--out") outPath = next("--out");
        else if (a == "--channel") channel = std::atoi(next("--channel").c_str());
        else if (a == "--rate") rate = std::atof(next("--rate").c_str());
        else if (a == "--block") block = std::atoi(next("--block").c_str());
        else if (a == "--instances") nInst = std::max(1, std::atoi(next("--instances").c_str()));
        else if (a == "--save-state") saveState = next("--save-state");
        else if (a == "--load-state") loadState = next("--load-state");
        else if (a == "--set") {
            const std::string kv = next("--set");
            const auto eq = kv.find('=');
            if (eq == std::string::npos) { std::cerr << "--set needs NAME=VALUE\n"; return 1; }
            sets.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        } else if (!a.empty() && a[0] != '-' && plugin.empty()) plugin = a;
        else { std::cerr << "unknown option " << a << "\n\n" << kUsage; return 1; }
    }

    auto host = owned(new HostApplication());
    PluginContextFactory::instance().setPluginContext(host);
    std::ostringstream quiet;
    PlugProvider::setErrorStream(&quiet);

    if (list) {
        const auto paths = VST3::Hosting::Module::getModulePaths();
        if (json) std::cout << "{\"plugins\":[";
        bool first = true;
        for (const auto& path : paths) {
            std::string err;
            auto mod = VST3::Hosting::Module::create(path, err);
            if (!mod) {
                if (!json) std::cout << path << "\n    could not load: " << err << "\n";
                continue;
            }
            for (const auto& ci : mod->getFactory().classInfos()) {
                if (ci.category() != kVstAudioEffectClass) continue;
                if (json) {
                    std::cout << (first ? "" : ",") << "{\"name\":\"" << jsonEsc(ci.name()) << "\",\"vendor\":\"" << jsonEsc(ci.vendor())
                              << "\",\"version\":\"" << jsonEsc(ci.version()) << "\",\"type\":\"" << jsonEsc(ci.subCategoriesString())
                              << "\",\"path\":\"" << jsonEsc(path) << "\"}";
                    first = false;
                } else {
                    std::cout << ci.vendor() << " " << ci.name() << "  " << ci.version() << "  (" << ci.subCategoriesString() << ")\n    "
                              << path << "\n";
                }
            }
        }
        if (json) std::cout << "]}\n";
        else if (paths.empty()) std::cout << "No VST3 plugins found in the standard folders.\n";
        return 0;
    }
    if (plugin.empty()) { std::cerr << kUsage; return 1; }

    std::string err;
    auto mod = VST3::Hosting::Module::create(plugin, err);
    if (!mod) { std::cerr << "Could not load " << plugin << ": " << err << "\n"; return 2; }
    const auto& factory = mod->getFactory();
    VST3::Hosting::ClassInfo ci;
    bool found = false;
    for (const auto& c : factory.classInfos()) {
        if (c.category() != kVstAudioEffectClass) continue;
        if (cls.empty() || lower(c.name()) == lower(cls)) { ci = c; found = true; break; }
    }
    if (!found) { std::cerr << "No audio effect" << (cls.empty() ? "" : " called \"" + cls + "\"") << " in " << plugin << "\n"; return 2; }

    // ---- input
    std::vector<float> input;
    std::string inputDesc;
    if (!inPath.empty()) {
        wav::Audio a;
        if (!wav::read(inPath, a, err)) { std::cerr << err << "\n"; return 1; }
        if (channel < 1 || channel > a.channels) { std::cerr << "--channel must be 1.." << a.channels << "\n"; return 1; }
        input.resize(a.frames());
        for (size_t i = 0; i < a.frames(); ++i) input[i] = a.samples[i * a.channels + channel - 1];
        rate = a.rate;
        inputDesc = inPath + " channel " + std::to_string(channel);
    } else {
        input = testSignal(rate, 10.0);
        inputDesc = "10 s test signal";
    }
    const size_t nBlocks = (input.size() + block - 1) / block;
    input.resize(nBlocks * block, 0.f);

    State loaded;
    if (!loadState.empty() && !loadStateFile(loadState, loaded)) { std::cerr << "Could not read settings file " << loadState << "\n"; return 1; }

    // ---- instance A: info, controls, the processed file
    Instance A;
    if (!A.open(factory, ci, rate, block, loadState.empty() ? nullptr : &loaded, err)) {
        std::cerr << ci.vendor() << " " << ci.name() << ": " << err << "\n" << quiet.str();
        return 3;
    }
    ParameterChanges changes;
    for (const auto& [name, value] : sets) {
        ParamID id; double v;
        if (!resolveSet(A.ctrl, name, value, id, v, err)) { std::cerr << err << "\n"; return 1; }
        A.ctrl->setParamNormalized(id, v);
        int32 qi = 0, pi = 0;
        if (auto* q = changes.addParameterData(id, qi)) q->addPoint(0, v, pi);
    }
    std::vector<float> out(input.size());
    bool finite = true, processedOk = true;
    for (size_t b = 0; b < nBlocks; ++b) {
        processedOk &= A.process(&input[b * block], &out[b * block], block, b == 0 ? &changes : nullptr);
    }
    for (float v : out) if (!std::isfinite(v)) { finite = false; break; }
    const auto ps = params(A.ctrl);
    if (!outPath.empty()) {
        std::vector<float> w = out;
        if (compensate && A.latency > 0) w.erase(w.begin(), w.begin() + std::min<size_t>(A.latency, w.size()));
        if (!wav::writeMonoFloat(outPath, w, static_cast<int>(rate))) { std::cerr << "Could not write " << outPath << "\n"; return 1; }
    }

    // ---- settings: save, restore into two fresh copies, compare
    State st;
    const bool gotState = A.getState(st);
    if (gotState && !saveState.empty() && !saveStateFile(saveState, st)) { std::cerr << "Could not write " << saveState << "\n"; return 1; }
    bool restoreOk = false, paramsMatch = false;
    double restoreDiffDb = -200.0;
    if (gotState) {
        Instance B, C;
        std::string e1, e2;
        if (B.open(factory, ci, rate, block, &st, e1) && C.open(factory, ci, rate, block, &st, e2)) {
            restoreOk = true;
            const auto pb = params(B.ctrl);
            paramsMatch = pb.size() == ps.size();
            for (size_t i = 0; paramsMatch && i < ps.size(); ++i) paramsMatch = std::fabs(pb[i].normalized - ps[i].normalized) < 1e-6;
            std::vector<float> ob(input.size()), oc(input.size());
            for (size_t b = 0; b < nBlocks; ++b) {
                B.process(&input[b * block], &ob[b * block], block, nullptr);
                C.process(&input[b * block], &oc[b * block], block, nullptr);
            }
            std::vector<float> d(input.size());
            for (size_t i = 0; i < d.size(); ++i) d[i] = ob[i] - oc[i];
            restoreDiffDb = peakDb(d);
        } else {
            err = e1.empty() ? e2 : e1;
        }
    }

    // ---- CPU: N instances processed together, block by block, as the engine would
    std::vector<std::unique_ptr<Instance>> many;
    for (int32 k = 0; k < nInst; ++k) {
        auto x = std::make_unique<Instance>();
        if (!x->open(factory, ci, rate, block, gotState ? &st : nullptr, err)) { std::cerr << "instance " << k + 1 << ": " << err << "\n"; return 3; }
        many.push_back(std::move(x));
    }
    std::vector<float> scratch(block);
    double total = 0, worst = 0;
    for (size_t b = 0; b < nBlocks; ++b) {
        const auto t0 = std::chrono::steady_clock::now();
        for (auto& x : many) x->process(&input[b * block], scratch.data(), block, nullptr);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        total += ms; worst = std::max(worst, ms);
    }
    many.clear();
    const double budgetMs = 1000.0 * block / rate;
    const double perInst = total / nBlocks / nInst;

    // ---- report
    const std::string latTxt = std::to_string(A.latency) + " samples (" + [&] { char b[32]; std::snprintf(b, sizeof b, "%.2f", 1000.0 * A.latency / rate); return std::string(b); }() + " ms)";
    if (json) {
        std::cout << "{\"plugin\":\"" << jsonEsc(ci.name()) << "\",\"vendor\":\"" << jsonEsc(ci.vendor()) << "\",\"version\":\"" << jsonEsc(ci.version())
                  << "\",\"type\":\"" << jsonEsc(ci.subCategoriesString()) << "\",\"path\":\"" << jsonEsc(plugin) << "\",\"rate\":" << rate
                  << ",\"block\":" << block << ",\"mono_native\":" << (A.monoNative ? "true" : "false") << ",\"in_channels\":" << A.inCh
                  << ",\"out_channels\":" << A.outCh << ",\"latency_samples\":" << A.latency << ",\"tail_samples\":" << A.tail
                  << ",\"processed_ok\":" << (processedOk ? "true" : "false") << ",\"output_finite\":" << (finite ? "true" : "false")
                  << ",\"output_peak_db\":" << peakDb(out) << ",\"input\":\"" << jsonEsc(inputDesc) << "\",\"state_saved\":" << (gotState ? "true" : "false")
                  << ",\"state_bytes\":" << st.component.size() + st.controller.size() << ",\"state_restores\":" << (restoreOk ? "true" : "false")
                  << ",\"state_params_match\":" << (paramsMatch ? "true" : "false") << ",\"restored_copies_diff_db\":" << restoreDiffDb
                  << ",\"instances\":" << nInst << ",\"cpu_ms_per_instance_per_block\":" << perInst << ",\"cpu_worst_block_ms\":" << worst
                  << ",\"block_budget_ms\":" << budgetMs << ",\"controls\":[";
        bool first = true;
        for (const auto& p : ps) {
            if (p.hidden) continue;
            std::cout << (first ? "" : ",") << "{\"id\":" << p.id << ",\"name\":\"" << jsonEsc(p.title) << "\",\"value\":\"" << jsonEsc(p.value)
                      << "\",\"units\":\"" << jsonEsc(p.units) << "\",\"normalized\":" << p.normalized << ",\"steps\":" << p.steps << "}";
            first = false;
        }
        std::cout << "]}\n";
    } else {
        std::printf("%s %s %s  (%s)\n", ci.vendor().c_str(), ci.name().c_str(), ci.version().c_str(), ci.subCategoriesString().c_str());
        std::printf("  %s\n\n", plugin.c_str());
        std::printf("Insert      %s\n", A.monoNative ? "runs as a mono insert" : ("no mono mode: Sotto feeds the same signal to all " + std::to_string(A.inCh) + " inputs and uses output 1").c_str());
        std::printf("Latency     %s at %.0f Hz, block %d%s\n", latTxt.c_str(), rate, block, A.tail ? (", tail " + std::to_string(A.tail) + " samples").c_str() : "");
        std::printf("Audio       %s, output peak %.1f dBFS%s\n", processedOk && finite ? "processed cleanly" : "PROBLEM: process failed or returned invalid samples",
                    peakDb(out), outPath.empty() ? "" : (", written to " + outPath).c_str());
        std::printf("Input       %s\n", inputDesc.c_str());
        if (!gotState) std::printf("Settings    PROBLEM: the plugin would not hand over its settings\n");
        else if (!restoreOk) std::printf("Settings    PROBLEM: saved (%zu bytes) but would not load back: %s\n", st.component.size() + st.controller.size(), err.c_str());
        else std::printf("Settings    save and restore work (%zu bytes)%s; two restored copies %s\n", st.component.size() + st.controller.size(),
                         paramsMatch ? ", controls match" : ", but some controls came back different",
                         restoreDiffDb < -120 ? "sound identical" : ("differ by up to " + std::to_string(static_cast<int>(restoreDiffDb)) + " dBFS (normal for adaptive plugins)").c_str());
        std::printf("CPU         %d instances: %.3f ms each per %.0f ms block (%.1f%% of one core each), worst block %.2f ms for all %d\n",
                    nInst, perInst, budgetMs, 100.0 * perInst / budgetMs, worst, nInst);
        std::printf("\nControls\n");
        for (const auto& p : ps) {
            if (p.hidden) continue;
            std::printf("  %-28s %s%s%s\n", p.title.c_str(), p.value.c_str(), p.units.empty() ? "" : " ", p.units.c_str());
        }
        if (!quiet.str().empty()) std::printf("\nPlugin messages\n%s", quiet.str().c_str());
    }
    A.close();
    if (!processedOk || !finite) return 3;
    if (!gotState || !restoreOk) return 4;
    return 0;
}
