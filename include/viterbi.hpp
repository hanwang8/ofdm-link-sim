// Rate 1/n convolutional encoder and sliding-window Viterbi decoder.
//
// Conventions match MATLAB poly2trellis / convenc / vitdec(..., 'trunc', ...):
//   - generators are written in octal, the most significant bit taps the current input;
//   - the encoder starts in the all-zero state and is not terminated;
//   - soft input: positive means bit 0, negative means bit 1, magnitude is confidence.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace viterbi {

using Bit = std::uint8_t;

class ConvolutionalCode {
public:
    // Example: ConvolutionalCode(7, {0133, 0171}) is the IEEE 802.11a code.
    ConvolutionalCode(int constraintLength, std::vector<unsigned> generators);

    int constraintLength() const { return k_; }
    int numOutputs() const { return static_cast<int>(generators_.size()); }
    unsigned numStates() const { return 1u << (k_ - 1); }

    // Output pattern for a transition; the first generator is the most significant bit.
    unsigned output(unsigned state, Bit input) const { return outputs_[2 * state + input]; }

    // The state's most significant bit holds the most recent input.
    unsigned nextState(unsigned state, Bit input) const {
        return (static_cast<unsigned>(input) << (k_ - 2)) | (state >> 1);
    }

    std::vector<Bit> encode(const std::vector<Bit>& bits) const;

private:
    int k_;
    std::vector<unsigned> generators_;
    std::vector<std::uint8_t> outputs_;
};

class ViterbiDecoder {
public:
    ViterbiDecoder(const ConvolutionalCode& code, int tracebackDepth);

    // Both return one decoded bit per n coded values, with the decoding delay removed.
    std::vector<Bit> decodeHard(const std::vector<Bit>& coded) const;
    std::vector<Bit> decodeSoft(const std::vector<double>& soft) const;

private:
    template <typename Metric, typename FillBranchMetrics>
    std::vector<Bit> decode(std::size_t steps, FillBranchMetrics fill) const;

    ConvolutionalCode code_;
    int depth_;
};

}  // namespace viterbi
