function cfg = ofdm_config()
%OFDM_CONFIG Link parameters shared by the simulation and vector export.
cfg.Nc = 64;                      % subcarriers
cfg.Ncp = 16;                     % cyclic prefix samples
cfg.M = 4;                        % QPSK
cfg.k = log2(cfg.M);              % bits per symbol
cfg.R = 1/2;                      % code rate
cfg.trellis = poly2trellis(7, [133 171]);
cfg.tb = 32;                      % Viterbi traceback depth
cfg.Nsym = 20000;                 % OFDM symbols per Eb/N0 point
cfg.pilotIdx = 1:4:cfg.Nc;        % 16 comb pilots, spacing 4; resolves up to 16 taps
cfg.hStatic = [1, 0.5, 0.2];      % static multipath taps
cfg.varFading = [0.7, 0.2, 0.1];  % fading tap variances (block fading per OFDM symbol)
end
