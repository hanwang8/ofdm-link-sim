// Measures decoder throughput and the BPSK/AWGN BER of the (133,171) code.
// Usage: viterbi_bench [bits_per_point]

#include "viterbi.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using viterbi::Bit;
using viterbi::ConvolutionalCode;
using viterbi::ViterbiDecoder;

namespace {

struct Link {
    std::vector<Bit> bits;
    std::vector<double> received;  // +1 for bit 0, -1 for bit 1, plus noise
    std::vector<Bit> hard;
};

Link makeLink(const ConvolutionalCode& code, std::size_t count, double ebN0Db, std::mt19937_64& rng) {
    Link link;
    link.bits.resize(count);
    for (Bit& b : link.bits) {
        b = static_cast<Bit>(rng() & 1u);
    }
    const std::vector<Bit> coded = code.encode(link.bits);

    const double rate = 1.0 / code.numOutputs();
    const double sigma = std::sqrt(1.0 / (2.0 * rate * std::pow(10.0, ebN0Db / 10.0)));
    std::normal_distribution<double> noise(0.0, sigma);
    link.received.resize(coded.size());
    link.hard.resize(coded.size());
    for (std::size_t i = 0; i < coded.size(); ++i) {
        link.received[i] = (coded[i] ? -1.0 : 1.0) + noise(rng);
        link.hard[i] = link.received[i] < 0.0 ? 1 : 0;
    }
    return link;
}

double ber(const std::vector<Bit>& decoded, const std::vector<Bit>& bits) {
    std::size_t errors = 0;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        errors += decoded[i] != bits[i];
    }
    return static_cast<double>(errors) / static_cast<double>(bits.size());
}

template <typename Decode>
double megabitsPerSecond(std::size_t count, Decode decode) {
    const auto start = std::chrono::steady_clock::now();
    const std::vector<Bit> decoded = decode();
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    if (decoded.size() != count) {
        std::abort();
    }
    return static_cast<double>(count) / elapsed.count() / 1e6;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t bitsPerPoint =
        argc > 1 ? static_cast<std::size_t>(std::atoll(argv[1])) : std::size_t{5000000};

    const ConvolutionalCode code(7, {0133, 0171});
    const ViterbiDecoder decoder(code, 32);
    std::mt19937_64 rng(2026);

    // Throughput: three runs, keep the best to reduce scheduling noise.
    const Link timing = makeLink(code, 2000000, 4.0, rng);
    double hardRate = 0.0;
    double softRate = 0.0;
    for (int run = 0; run < 3; ++run) {
        hardRate = std::max(hardRate, megabitsPerSecond(timing.bits.size(), [&] { return decoder.decodeHard(timing.hard); }));
        softRate = std::max(softRate, megabitsPerSecond(timing.bits.size(), [&] { return decoder.decodeSoft(timing.received); }));
    }
    std::printf("Throughput, K=7, traceback 32, one thread: hard %.1f Mbit/s, soft %.1f Mbit/s\n\n", hardRate, softRate);

    std::printf("EbN0_dB,uncoded_bpsk,hard,soft\n");
    for (double ebN0Db = 0.0; ebN0Db <= 7.0; ebN0Db += 0.5) {
        const Link link = makeLink(code, bitsPerPoint, ebN0Db, rng);
        const double uncoded = 0.5 * std::erfc(std::sqrt(std::pow(10.0, ebN0Db / 10.0)));
        std::printf("%.1f,%.3e,%.3e,%.3e\n", ebN0Db, uncoded, ber(decoder.decodeHard(link.hard), link.bits),
                    ber(decoder.decodeSoft(link.received), link.bits));
    }
    return 0;
}
