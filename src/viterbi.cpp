#include "viterbi.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace viterbi {

namespace {

unsigned parity(unsigned value) {
    unsigned p = 0;
    while (value != 0) {
        p ^= value & 1u;
        value >>= 1;
    }
    return p;
}

}  // namespace

ConvolutionalCode::ConvolutionalCode(int constraintLength, std::vector<unsigned> generators)
    : k_(constraintLength), generators_(std::move(generators)) {
    if (k_ < 2 || k_ > 16) {
        throw std::invalid_argument("constraint length must be between 2 and 16");
    }
    if (generators_.empty() || generators_.size() > 8) {
        throw std::invalid_argument("number of generators must be between 1 and 8");
    }
    for (unsigned g : generators_) {
        if (g == 0 || g >= (1u << k_)) {
            throw std::invalid_argument("generator does not fit the constraint length");
        }
    }

    // Each transition is one register value: the input on top of the current state.
    outputs_.resize(2 * numStates());
    for (unsigned state = 0; state < numStates(); ++state) {
        for (unsigned input = 0; input < 2; ++input) {
            const unsigned reg = (input << (k_ - 1)) | state;
            unsigned pattern = 0;
            for (unsigned g : generators_) {
                pattern = (pattern << 1) | parity(reg & g);
            }
            outputs_[2 * state + input] = static_cast<std::uint8_t>(pattern);
        }
    }
}

std::vector<Bit> ConvolutionalCode::encode(const std::vector<Bit>& bits) const {
    const int n = numOutputs();
    std::vector<Bit> coded;
    coded.reserve(bits.size() * n);
    unsigned state = 0;
    for (Bit bit : bits) {
        const Bit input = bit ? 1 : 0;
        const unsigned pattern = output(state, input);
        for (int i = n - 1; i >= 0; --i) {
            coded.push_back(static_cast<Bit>((pattern >> i) & 1u));
        }
        state = nextState(state, input);
    }
    return coded;
}

ViterbiDecoder::ViterbiDecoder(const ConvolutionalCode& code, int tracebackDepth)
    : code_(code), depth_(tracebackDepth) {
    if (depth_ < 1) {
        throw std::invalid_argument("traceback depth must be positive");
    }
}

template <typename Metric, typename FillBranchMetrics>
std::vector<Bit> ViterbiDecoder::decode(std::size_t steps, FillBranchMetrics fill) const {
    const unsigned numStates = code_.numStates();
    const unsigned mask = numStates - 1;
    const int msbShift = code_.constraintLength() - 2;
    const std::size_t depth = static_cast<std::size_t>(depth_);

    // Every state but the all-zero start is unreachable at time 0.
    const Metric unreachable = std::numeric_limits<Metric>::max() / 2;
    const Metric renormaliseAbove = std::numeric_limits<Metric>::max() / 4;
    std::vector<Metric> current(numStates, unreachable);
    std::vector<Metric> next(numStates);
    current[0] = 0;

    std::vector<Metric> branch(std::size_t{1} << code_.numOutputs());
    // Ring buffer of survivors: which predecessor (0 or 1) each state came from.
    std::vector<std::uint8_t> survivors(depth * numStates);

    std::vector<Bit> decoded(steps, 0);

    // Walks back from `state` at the end of step `from`, undoing `count` steps.
    const auto walkBack = [&](unsigned state, std::size_t from, std::size_t count) {
        std::size_t slot = from % depth;
        for (std::size_t i = 0; i < count; ++i) {
            state = ((state << 1) & mask) | survivors[slot * numStates + state];
            slot = (slot == 0) ? depth - 1 : slot - 1;
        }
        return state;
    };

    unsigned best = 0;
    for (std::size_t t = 0; t < steps; ++t) {
        fill(t, branch);

        // Add-compare-select. A state has two predecessors differing only in their last bit,
        // and both reach it with the same input: the state's most significant bit.
        std::uint8_t* slot = &survivors[(t % depth) * numStates];
        for (unsigned s = 0; s < numStates; ++s) {
            const Bit input = static_cast<Bit>(s >> msbShift);
            const unsigned p0 = (s << 1) & mask;
            const unsigned p1 = p0 | 1u;
            const Metric m0 = current[p0] + branch[code_.output(p0, input)];
            const Metric m1 = current[p1] + branch[code_.output(p1, input)];
            const bool takeP1 = m1 < m0;
            next[s] = takeP1 ? m1 : m0;
            slot[s] = takeP1 ? 1 : 0;
        }
        current.swap(next);

        // Ties go to the highest-numbered state. Any choice decodes equally well on average;
        // this one, with the lower predecessor winning above, reproduces MATLAB vitdec bit for bit.
        best = 0;
        for (unsigned s = 1; s < numStates; ++s) {
            if (current[s] <= current[best]) {
                best = s;
            }
        }

        // Only differences between metrics matter, so shift them back before they overflow.
        const Metric bestMetric = current[best];
        if (bestMetric > renormaliseAbove || bestMetric < -renormaliseAbove) {
            for (Metric& m : current) {
                m -= bestMetric;
            }
        }

        if (t >= depth) {
            decoded[t - depth] = static_cast<Bit>(walkBack(best, t, depth) >> msbShift);
        }
    }

    // End of input: the remaining bits all come from the final best state.
    const std::size_t remaining = std::min(depth, steps);
    for (std::size_t i = 0; i < remaining; ++i) {
        const unsigned state = walkBack(best, steps - 1, i);
        decoded[steps - 1 - i] = static_cast<Bit>(state >> msbShift);
    }
    return decoded;
}

std::vector<Bit> ViterbiDecoder::decodeHard(const std::vector<Bit>& coded) const {
    const std::size_t n = static_cast<std::size_t>(code_.numOutputs());
    if (coded.size() % n != 0) {
        throw std::invalid_argument("coded length must be a multiple of the number of outputs");
    }
    // Hamming distance from the received bits to each possible output pattern.
    return decode<std::int32_t>(coded.size() / n, [&](std::size_t t, std::vector<std::int32_t>& branch) {
        for (std::size_t pattern = 0; pattern < branch.size(); ++pattern) {
            std::int32_t distance = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const unsigned expected = (pattern >> (n - 1 - i)) & 1u;
                distance += (expected != (coded[t * n + i] ? 1u : 0u)) ? 1 : 0;
            }
            branch[pattern] = distance;
        }
    });
}

std::vector<Bit> ViterbiDecoder::decodeSoft(const std::vector<double>& soft) const {
    const std::size_t n = static_cast<std::size_t>(code_.numOutputs());
    if (soft.size() % n != 0) {
        throw std::invalid_argument("soft length must be a multiple of the number of outputs");
    }
    // Negative correlation with the expected +1 (bit 0) / -1 (bit 1) values. This equals the
    // squared Euclidean distance up to terms shared by all patterns, so smaller is better.
    return decode<double>(soft.size() / n, [&](std::size_t t, std::vector<double>& branch) {
        for (std::size_t pattern = 0; pattern < branch.size(); ++pattern) {
            double metric = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const unsigned expected = (pattern >> (n - 1 - i)) & 1u;
                metric += expected ? soft[t * n + i] : -soft[t * n + i];
            }
            branch[pattern] = metric;
        }
    });
}

}  // namespace viterbi
