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
#include "host_core.h"
#include "wav.h"

using namespace Steinberg;
using namespace Steinberg::Vst;
using namespace sotto;

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
