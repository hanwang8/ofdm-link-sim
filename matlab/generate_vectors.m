% Exports inputs and MATLAB convenc / vitdec outputs for the C++ tests, then times vitdec.
% Run from this folder:  generate_vectors
% Writes ../tests/vectors/*.txt

rng(20261106);
cfg = ofdm_config();
trellis = cfg.trellis;
outDir = fullfile('..', 'tests', 'vectors');
if ~exist(outDir, 'dir'); mkdir(outDir); end

N = 20000;
bits = randi([0 1], N, 1);
coded = convenc(bits, trellis);

% Encoder
writeVector(outDir, 'encode', 'encode', 32, bits, coded, '%d');

% Hard decisions over a binary symmetric channel, with odd and even traceback depths
for c = {{'hard_p02_tb32', 0.02, 32}, {'hard_p05_tb32', 0.05, 32}, {'hard_p10_tb32', 0.10, 32}, ...
         {'hard_p05_tb16', 0.05, 16}, {'hard_p05_tb35', 0.05, 35}}
    [name, p, tb] = deal(c{1}{:});
    rx = double(xor(coded, rand(size(coded)) < p));
    writeVector(outDir, name, 'hard', tb, rx, vitdec(rx, trellis, tb, 'trunc', 'hard'), '%d');
end

% Soft decisions over AWGN
for c = {{'soft_s06_tb32', 0.6, 32}, {'soft_s08_tb32', 0.8, 32}, {'soft_s10_tb32', 1.0, 32}, {'soft_s08_tb16', 0.8, 16}}
    [name, sigma, tb] = deal(c{1}{:});
    rx = (1 - 2 * coded) + sigma * randn(size(coded));
    writeVector(outDir, name, 'soft', tb, rx, vitdec(rx, trellis, tb, 'trunc', 'unquant'), '%.17g');
end

% Soft values taken from the real OFDM receiver, so the C++ decoder sees the link's own data
small = cfg; small.Nsym = 400;
for c = {{'ofdm_fading_6dB', 'fading', 6}, {'ofdm_static_6dB', 'static', 6}}
    [name, ch, ebn0] = deal(c{1}{:});
    r = ofdm_link_point(small, cfg.pilotIdx, ch, ebn0);
    writeVector(outDir, name, 'soft', cfg.tb, r.softLmmse, ...
        vitdec(r.softLmmse, trellis, cfg.tb, 'trunc', 'unquant'), '%.17g');
end
fprintf('Vectors written to %s\n', outDir);

% vitdec throughput on the same workload as tools/viterbi_bench.cpp
M = 2e6;
bits = randi([0 1], M, 1);
coded = convenc(bits, trellis);
sigma = sqrt(1 / (2 * cfg.R * 10^(4 / 10)));
soft = (1 - 2 * coded) + sigma * randn(size(coded));
hard = double(soft < 0);
hardTime = inf; softTime = inf;
for run = 1:3
    tic; vitdec(hard, trellis, 32, 'trunc', 'hard'); hardTime = min(hardTime, toc);
    tic; vitdec(soft, trellis, 32, 'trunc', 'unquant'); softTime = min(softTime, toc);
end
fprintf('vitdec throughput: hard %.1f Mbit/s, soft %.1f Mbit/s\n', M / hardTime / 1e6, M / softTime / 1e6);

function writeVector(outDir, name, mode, tb, input, expected, fmt)
fid = fopen(fullfile(outDir, [name '.txt']), 'w');
fprintf(fid, '%s %d\n', mode, tb);
fprintf(fid, [fmt ' '], input);
fprintf(fid, '\n');
fprintf(fid, '%d ', expected);
fprintf(fid, '\n');
fclose(fid);
end
