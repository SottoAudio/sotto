// Shared VST3 hosting code for sotto-plugin-test and the sotto_vst3host library: one plugin
// instance used as a mono insert, its controls, and saving/restoring its settings.
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

namespace sotto {
using namespace Steinberg;
using namespace Steinberg::Vst;

inline std::string u8(const TChar* s) {
    return VST3::StringConvert::convert(std::u16string(reinterpret_cast<const char16_t*>(s)));
}

inline std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

struct State {
    std::vector<char> component, controller;
    bool empty() const { return component.empty(); }
};

inline bool saveStateFile(const std::string& path, const State& s) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint64_t a = s.component.size(), b = s.controller.size();
    f.write("SOTTOVST3STATE01", 16);
    f.write(reinterpret_cast<const char*>(&a), 8); f.write(s.component.data(), static_cast<std::streamsize>(a));
    f.write(reinterpret_cast<const char*>(&b), 8); f.write(s.controller.data(), static_cast<std::streamsize>(b));
    return static_cast<bool>(f);
}

inline bool loadStateFile(const std::string& path, State& s) {
    std::ifstream f(path, std::ios::binary);
    char magic[16];
    if (!f.read(magic, 16) || std::memcmp(magic, "SOTTOVST3STATE01", 16)) return false;
    uint64_t a = 0, b = 0;
    f.read(reinterpret_cast<char*>(&a), 8); s.component.resize(a); f.read(s.component.data(), static_cast<std::streamsize>(a));
    f.read(reinterpret_cast<char*>(&b), 8); s.controller.resize(b); f.read(s.controller.data(), static_cast<std::streamsize>(b));
    return static_cast<bool>(f);
}

inline IPtr<MemoryStream> streamOf(const std::vector<char>& bytes) {
    auto s = owned(new MemoryStream());
    int32 written = 0;
    if (!bytes.empty()) s->write(const_cast<char*>(bytes.data()), static_cast<int32>(bytes.size()), &written);
    s->seek(0, IBStream::kIBSeekSet, nullptr);
    return s;
}

inline std::vector<char> bytesOf(MemoryStream& s) {
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

inline std::vector<Param> params(IEditController* ctrl) {
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
inline bool resolveSet(IEditController* ctrl, const std::string& name, const std::string& value, ParamID& id, double& norm,
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

}  // namespace sotto
