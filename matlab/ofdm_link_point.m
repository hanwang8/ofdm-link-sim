function out = ofdm_link_point(cfg, pilotIdx, channel, EbN0_dB)
%OFDM_LINK_POINT Simulate one Eb/N0 point of the OFDM link.
%   pilotIdx = [] gives a pilot-free frame, decoded with the true channel only.
%   Otherwise the channel is also estimated from the pilots with LS and LMMSE.
%   Every receiver is decoded with both hard- and soft-decision Viterbi.

Nc = cfg.Nc; Ncp = cfg.Ncp; Nsym = cfg.Nsym;
dataIdx = setdiff(1:Nc, pilotIdx);
Nd = numel(dataIdx);

% Transmitter
dataBits = randi([0 1], Nsym * Nd * cfg.k * cfg.R, 1);
coded = convenc(dataBits, cfg.trellis);
symIdx = 2 * coded(1:2:end) + coded(2:2:end);
X = zeros(Nc, Nsym);
X(dataIdx, :) = reshape(pskmod(symIdx, cfg.M, pi/4), Nd, Nsym);
pilotSym = exp(1j * pi/4) * ones(numel(pilotIdx), 1);
X(pilotIdx, :) = repmat(pilotSym, 1, Nsym);

x = ifft(X, Nc);
xCp = [x(Nc - Ncp + 1:Nc, :); x];

% Channel: taps (L x Nsym) and true frequency response (Nc x Nsym)
if strcmp(channel, 'static')
    taps = repmat(cfg.hStatic(:), 1, Nsym);
    pdp = abs(cfg.hStatic(:)).^2;
else
    pdp = cfg.varFading(:);
    taps = sqrt(pdp / 2) .* (randn(numel(pdp), Nsym) + 1j * randn(numel(pdp), Nsym));
end
H = fft(taps, Nc, 1);
y = zeros(size(xCp));
for s = 1:Nsym
    y(:, s) = filter(taps(:, s), 1, xCp(:, s));
end

% AWGN, referenced to measured receive power, matching awgn(..., 'measured').
% Eb/N0 counts only information bits, so pilots and CP are charged as overhead.
snrDb = EbN0_dB + 10*log10(cfg.k * cfg.R) + 10*log10(Nd / Nc) + 10*log10(Nc / (Nc + Ncp));
snr = 10^(snrDb / 10);
rxPower = mean(abs(y(:)).^2);
noiseVar = rxPower / snr;
y = y + sqrt(noiseVar / 2) * (randn(size(y)) + 1j * randn(size(y)));

% Receiver front end
Y = fft(y(Ncp + 1:end, :), Nc, 1);
[out.berKnownHard, out.berKnownSoft] = decodeBoth(Y, H, dataIdx, dataBits, cfg);
if isempty(pilotIdx); return; end

% Noise variance per subcarrier after the unnormalised FFT
noiseVarF = Nc * noiseVar;

% LS at pilots
HpLs = Y(pilotIdx, :) ./ pilotSym;

% LS + linear interpolation across frequency (H is periodic in k, so wrap)
kp = [pilotIdx, pilotIdx(1) + Nc];
HLs = interp1(kp, [HpLs; HpLs(1, :)], (1:Nc).', 'linear');

% LMMSE from the pilot LS estimates, using the known PDP and noise power
F = exp(-2j * pi * (0:Nc - 1).' * (0:numel(pdp) - 1) / Nc);
R = F * diag(pdp) * F';
W = R(:, pilotIdx) / (R(pilotIdx, pilotIdx) + noiseVarF * eye(numel(pilotIdx)));
HLmmse = W * HpLs;

[out.berLsHard, out.berLsSoft] = decodeBoth(Y, HLs, dataIdx, dataBits, cfg);
[out.berLmmseHard, out.berLmmseSoft, out.softLmmse] = decodeBoth(Y, HLmmse, dataIdx, dataBits, cfg);
out.nmseLs = nmseDb(HLs, H);
out.nmseLmmse = nmseDb(HLmmse, H);
out.nmseLsPilots = nmseDb(HpLs, H(pilotIdx, :));
out.expectedLsPilotsDb = -10*log10(snr);
end

function [berHard, berSoft, soft] = decodeBoth(Y, Hhat, dataIdx, dataBits, cfg)
% One-tap zero-forcing, then hard or soft demapping and Viterbi decoding.
eq = Y(dataIdx, :) ./ Hhat(dataIdx, :);

sym = pskdemod(eq(:), cfg.M, pi/4);
bits = reshape([floor(sym / 2), mod(sym, 2)].', [], 1);
berHard = mean(vitdec(bits, cfg.trellis, cfg.tb, 'trunc', 'hard') ~= dataBits);

% QPSK Gray map: the first bit is 1 when Im < 0, the second when Re < 0. ZF noise grows as
% 1/|H|^2 and varies by subcarrier, so each value is weighted by |H|^2 to act as an LLR.
llr = abs(Hhat(dataIdx, :)).^2 .* eq;
soft = zeros(2 * numel(llr), 1);
soft(1:2:end) = imag(llr(:));
soft(2:2:end) = real(llr(:));
berSoft = mean(vitdec(soft, cfg.trellis, cfg.tb, 'trunc', 'unquant') ~= dataBits);
end

function v = nmseDb(Hhat, H)
v = 10*log10(mean(abs(Hhat(:) - H(:)).^2) / mean(abs(H(:)).^2));
end
