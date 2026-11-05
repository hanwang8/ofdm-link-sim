// Unit tests, plus an exact comparison against vectors exported from MATLAB.
// Usage: test_viterbi [vector_directory]

#include "viterbi.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using viterbi::Bit;
using viterbi::ConvolutionalCode;
using viterbi::ViterbiDecoder;

namespace {

const ConvolutionalCode kCode(7, {0133, 0171});
constexpr int kDepth = 32;

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<Bit> randomBits(std::size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<Bit> bits(count);
    for (Bit& b : bits) {
        b = static_cast<Bit>(rng() & 1u);
    }
    return bits;
}

std::vector<double> toSoft(const std::vector<Bit>& coded, double amplitude) {
    std::vector<double> soft;
    soft.reserve(coded.size());
    for (Bit b : coded) {
        soft.push_back(b ? -amplitude : amplitude);
    }
    return soft;
}

void testImpulseResponse() {
    // A single 1 followed by zeros reads the generators out: 1011011 and 1111001.
    const std::vector<Bit> input = {1, 0, 0, 0, 0, 0, 0};
    const std::vector<Bit> expected = {1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1};
    check(kCode.encode(input) == expected, "impulse response does not match the generators");
}

void testRoundTripWithoutNoise() {
    for (std::size_t length : {std::size_t{1}, std::size_t{5}, std::size_t{31}, std::size_t{32},
                               std::size_t{33}, std::size_t{1000}}) {
        const std::vector<Bit> bits = randomBits(length, 11);
        const std::vector<Bit> coded = kCode.encode(bits);
        const ViterbiDecoder decoder(kCode, kDepth);
        check(decoder.decodeHard(coded) == bits, "hard round trip failed");
        check(decoder.decodeSoft(toSoft(coded, 1.0)) == bits, "soft round trip failed");
    }
}

void testCorrectsErrorsWithinFreeDistance() {
    // Free distance is 10, so four errors inside one constraint span must be corrected.
    const std::vector<Bit> bits = randomBits(400, 22);
    std::vector<Bit> coded = kCode.encode(bits);
    for (std::size_t position : {std::size_t{200}, std::size_t{203}, std::size_t{207}, std::size_t{210}}) {
        coded[position] ^= 1u;
    }
    check(ViterbiDecoder(kCode, kDepth).decodeHard(coded) == bits, "failed to correct four errors");
}

void testSoftUsesConfidence() {
    // A burst of flips marked as unreliable must be outvoted by the reliable bits around it.
    const std::vector<Bit> bits = randomBits(400, 33);
    const std::vector<Bit> coded = kCode.encode(bits);
    std::vector<double> soft = toSoft(coded, 1.0);
    for (std::size_t position = 200; position < 212; ++position) {
        soft[position] = -0.1 * soft[position];
    }
    // As hard decisions, twelve consecutive errors defeat the decoder.
    std::vector<Bit> hard;
    for (double s : soft) {
        hard.push_back(s < 0 ? 1 : 0);
    }
    const ViterbiDecoder decoder(kCode, kDepth);
    check(decoder.decodeHard(hard) != bits, "test setup should defeat the hard decoder");
    check(decoder.decodeSoft(soft) == bits, "soft decoder failed to use reliability");
}

void testSoftIgnoresScale() {
    const std::vector<Bit> bits = randomBits(2000, 44);
    const std::vector<Bit> coded = kCode.encode(bits);
    std::mt19937 rng(45);
    std::normal_distribution<double> noise(0.0, 0.8);
    std::vector<double> soft = toSoft(coded, 1.0);
    std::vector<double> scaled = soft;
    for (std::size_t i = 0; i < soft.size(); ++i) {
        soft[i] += noise(rng);
        scaled[i] = 8.0 * soft[i];
    }
    const ViterbiDecoder decoder(kCode, kDepth);
    check(decoder.decodeSoft(soft) == decoder.decodeSoft(scaled), "decisions depend on soft scale");
}

void testLongInputAtLowErrorRate() {
    const std::vector<Bit> bits = randomBits(200000, 55);
    std::vector<Bit> coded = kCode.encode(bits);
    std::mt19937 rng(56);
    std::bernoulli_distribution flip(0.01);
    for (Bit& b : coded) {
        b ^= flip(rng) ? 1u : 0u;
    }
    const std::vector<Bit> decoded = ViterbiDecoder(kCode, kDepth).decodeHard(coded);
    std::size_t errors = 0;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        errors += decoded[i] != bits[i];
    }
    check(errors < 20, "unexpectedly many errors at 1% channel error rate: " + std::to_string(errors));
}

void testMetricRenormalisation() {
    // Large soft values drive path metrics past the renormalisation threshold. Without it they
    // would overflow to infinity and every later decision would be meaningless.
    const std::vector<Bit> bits = randomBits(6000, 77);
    const std::vector<Bit> coded = kCode.encode(bits);
    std::mt19937 rng(78);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<double> soft = toSoft(coded, 1.0);
    std::vector<double> huge = soft;
    for (std::size_t i = 0; i < soft.size(); ++i) {
        soft[i] += noise(rng);
        huge[i] = 1e304 * soft[i];
    }
    const ViterbiDecoder decoder(kCode, kDepth);
    check(decoder.decodeSoft(huge) == decoder.decodeSoft(soft), "large soft values changed the decisions");
}

void testEdgeCases() {
    const ViterbiDecoder decoder(kCode, kDepth);
    check(decoder.decodeHard({}).empty(), "empty hard input should give empty output");
    check(decoder.decodeSoft({}).empty(), "empty soft input should give empty output");

    const auto throws = [](const std::function<void()>& action) {
        try {
            action();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check(throws([&] { decoder.decodeHard({1, 0, 1}); }), "odd hard length should throw");
    check(throws([&] { decoder.decodeSoft({1.0, 0.0, 1.0}); }), "odd soft length should throw");
    check(throws([] { ViterbiDecoder(kCode, 0); }), "zero depth should throw");
    check(throws([] { ConvolutionalCode(7, {}); }), "no generators should throw");
    check(throws([] { ConvolutionalCode(3, {0133}); }), "oversized generator should throw");
    check(throws([] { ConvolutionalCode(1, {1}); }), "constraint length 1 should throw");
}

void testOtherCodes() {
    // The decoder is generic: try a smaller code and a rate 1/3 code.
    const std::vector<Bit> bits = randomBits(500, 66);
    const ConvolutionalCode small(3, {07, 05});
    check(ViterbiDecoder(small, 15).decodeHard(small.encode(bits)) == bits, "K=3 round trip failed");
    const ConvolutionalCode third(7, {0133, 0145, 0175});
    check(ViterbiDecoder(third, 40).decodeHard(third.encode(bits)) == bits, "rate 1/3 round trip failed");
}

// Exact comparison with MATLAB. File format, written by matlab/generate_vectors.m:
//   line 1: mode (encode, hard or soft) and traceback depth
//   line 2: input values
//   line 3: MATLAB output bits
std::size_t runVectorFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    check(file.good(), "cannot open " + path.string());

    std::string mode;
    int depth = 0;
    file >> mode >> depth;
    std::string line;
    std::getline(file, line);

    std::getline(file, line);
    std::istringstream inputStream(line);
    std::vector<double> input;
    for (double value; inputStream >> value;) {
        input.push_back(value);
    }

    std::getline(file, line);
    std::istringstream expectedStream(line);
    std::vector<Bit> expected;
    for (int value; expectedStream >> value;) {
        expected.push_back(static_cast<Bit>(value));
    }
    check(!input.empty() && !expected.empty(), "empty vector file " + path.string());

    std::vector<Bit> actual;
    if (mode == "soft") {
        actual = ViterbiDecoder(kCode, depth).decodeSoft(input);
    } else {
        std::vector<Bit> bits;
        bits.reserve(input.size());
        for (double value : input) {
            bits.push_back(static_cast<Bit>(value));
        }
        actual = (mode == "encode") ? kCode.encode(bits) : ViterbiDecoder(kCode, depth).decodeHard(bits);
    }

    check(actual.size() == expected.size(), "length differs from MATLAB in " + path.filename().string());
    std::size_t differences = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        differences += actual[i] != expected[i];
    }
    std::printf("  %-28s %8zu bits, %zu differ from MATLAB\n", path.filename().string().c_str(),
                actual.size(), differences);
    return differences;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"impulse response", testImpulseResponse},
        {"round trip without noise", testRoundTripWithoutNoise},
        {"corrects errors within free distance", testCorrectsErrorsWithinFreeDistance},
        {"soft decoding uses confidence", testSoftUsesConfidence},
        {"soft decisions ignore scale", testSoftIgnoresScale},
        {"long input at low error rate", testLongInputAtLowErrorRate},
        {"metric renormalisation", testMetricRenormalisation},
        {"edge cases", testEdgeCases},
        {"other codes", testOtherCodes},
    };

    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::printf("PASS  %s\n", name.c_str());
        } catch (const std::exception& error) {
            std::printf("FAIL  %s: %s\n", name.c_str(), error.what());
            ++failures;
        }
    }

    if (argc > 1) {
        std::printf("MATLAB vectors in %s\n", argv[1]);
        std::size_t files = 0;
        std::size_t differences = 0;
        try {
            for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
                if (entry.path().extension() == ".txt") {
                    differences += runVectorFile(entry.path());
                    ++files;
                }
            }
            check(files > 0, "no vector files found");
            check(differences == 0, std::to_string(differences) + " bits differ from MATLAB");
            std::printf("PASS  matches MATLAB on %zu files\n", files);
        } catch (const std::exception& error) {
            std::printf("FAIL  MATLAB vectors: %s\n", error.what());
            ++failures;
        }
    }

    std::printf(failures == 0 ? "All tests passed.\n" : "%d test group(s) failed.\n", failures);
    return failures == 0 ? 0 : 1;
}
