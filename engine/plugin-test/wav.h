// Minimal WAV reading and writing for sotto-plugin-test: PCM 16/24/32-bit and 32-bit float in,
// 32-bit float out. No dependencies.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace wav {

struct Audio {
    int rate = 48000;
    int channels = 1;
    std::vector<float> samples;   // interleaved
    size_t frames() const { return channels ? samples.size() / channels : 0; }
};

inline bool read(const std::string& path, Audio& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    char riff[12];
    if (!f.read(riff, 12) || std::memcmp(riff, "RIFF", 4) || std::memcmp(riff + 8, "WAVE", 4)) {
        err = path + " is not a WAV file"; return false;
    }
    uint16_t fmt = 0, ch = 0, bits = 0; uint32_t rate = 0;
    std::vector<char> data;
    while (f) {
        char id[4]; uint32_t len = 0;
        if (!f.read(id, 4) || !f.read(reinterpret_cast<char*>(&len), 4)) break;
        if (!std::memcmp(id, "fmt ", 4)) {
            std::vector<char> b(len);
            f.read(b.data(), len);
            std::memcpy(&fmt, &b[0], 2); std::memcpy(&ch, &b[2], 2);
            std::memcpy(&rate, &b[4], 4); std::memcpy(&bits, &b[14], 2);
            if (fmt == 0xFFFE && len >= 26) std::memcpy(&fmt, &b[24], 2);   // WAVE_FORMAT_EXTENSIBLE
        } else if (!std::memcmp(id, "data", 4)) {
            data.resize(len);
            f.read(data.data(), len);
            data.resize(static_cast<size_t>(f.gcount()));
            break;
        } else {
            f.seekg(len + (len & 1), std::ios::cur);
        }
        if (len & 1) f.seekg(1, std::ios::cur);
    }
    if (!ch || !rate || data.empty()) { err = path + ": no audio found"; return false; }
    const int bps = bits / 8;
    const size_t n = data.size() / bps;
    out.rate = static_cast<int>(rate); out.channels = ch; out.samples.resize(n);
    const unsigned char* p = reinterpret_cast<const unsigned char*>(data.data());
    for (size_t i = 0; i < n; ++i, p += bps) {
        float v = 0.f;
        if (fmt == 3 && bits == 32) { std::memcpy(&v, p, 4); }
        else if (fmt == 1 && bits == 16) { int16_t s; std::memcpy(&s, p, 2); v = s / 32768.f; }
        else if (fmt == 1 && bits == 24) { int32_t s = (p[0] | (p[1] << 8) | (p[2] << 16)); if (s & 0x800000) s |= ~0xFFFFFF; v = s / 8388608.f; }
        else if (fmt == 1 && bits == 32) { int32_t s; std::memcpy(&s, p, 4); v = static_cast<float>(s / 2147483648.0); }
        else { err = path + ": unsupported WAV format (use 16/24/32-bit PCM or 32-bit float)"; return false; }
        out.samples[i] = v;
    }
    return true;
}

inline bool writeMonoFloat(const std::string& path, const std::vector<float>& x, int rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t dataLen = static_cast<uint32_t>(x.size() * 4), riffLen = 36 + dataLen;
    const uint16_t fmt = 3, ch = 1, bits = 32, align = 4; const uint32_t r = rate, bps = rate * 4, fmtLen = 16;
    f.write("RIFF", 4); f.write(reinterpret_cast<const char*>(&riffLen), 4); f.write("WAVE", 4);
    f.write("fmt ", 4); f.write(reinterpret_cast<const char*>(&fmtLen), 4);
    f.write(reinterpret_cast<const char*>(&fmt), 2); f.write(reinterpret_cast<const char*>(&ch), 2);
    f.write(reinterpret_cast<const char*>(&r), 4); f.write(reinterpret_cast<const char*>(&bps), 4);
    f.write(reinterpret_cast<const char*>(&align), 2); f.write(reinterpret_cast<const char*>(&bits), 2);
    f.write("data", 4); f.write(reinterpret_cast<const char*>(&dataLen), 4);
    f.write(reinterpret_cast<const char*>(x.data()), dataLen);
    return static_cast<bool>(f);
}

}  // namespace wav
