% Sweeps Eb/N0 for the OFDM link and compares channel estimators and Viterbi decision types.
% Run from this folder:  seed = 1; run_ofdm_link
% Writes ../results/ofdm_results_seed<seed>.csv and ../results/ofdm_ber_seed<seed>.png

if ~exist('seed', 'var'); seed = 1; end
rng(seed);
cfg = ofdm_config();
EbN0_dB = 0:2:20;
resultsDir = fullfile('..', 'results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

selfCheck(cfg);

%% Simulation
channels = {'static', 'fading'};
results = table();
for c = 1:2
    for i = 1:numel(EbN0_dB)
        a = ofdm_link_point(cfg, [], channels{c}, EbN0_dB(i));
        b = ofdm_link_point(cfg, cfg.pilotIdx, channels{c}, EbN0_dB(i));
        row = table(string(channels{c}), EbN0_dB(i), a.berKnownHard, a.berKnownSoft, ...
            b.berKnownHard, b.berKnownSoft, b.berLmmseHard, b.berLmmseSoft, b.berLsHard, b.berLsSoft, ...
            b.nmseLmmse, b.nmseLs, b.nmseLsPilots, ...
            'VariableNames', {'channel', 'EbN0_dB', 'known_no_pilots_hard', 'known_no_pilots_soft', ...
            'known_pilots_hard', 'known_pilots_soft', 'lmmse_hard', 'lmmse_soft', 'ls_hard', 'ls_soft', ...
            'nmse_lmmse_dB', 'nmse_ls_dB', 'nmse_ls_at_pilots_dB'});
        results = [results; row]; %#ok<AGROW>
    end
end
disp(results);
writetable(results, fullfile(resultsDir, sprintf('ofdm_results_seed%d.csv', seed)));

%% Figure: each receiver with hard (dashed) and soft (solid) decisions
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1000 430]);
tiledlayout(fig, 1, 2);
names = {'known_pilots', 'lmmse', 'ls'};
labels = {'Known H', 'LMMSE', 'LS + linear'};
colours = {'k', 'b', 'r'};
markers = {'o', 's', '^'};
for c = 1:2
    nexttile;
    t = results(results.channel == channels{c}, :);
    for n = 1:3
        semilogy(t.EbN0_dB, max(t.([names{n} '_soft']), 1e-7), [colours{n} '-' markers{n}], ...
            'LineWidth', 1.5, 'DisplayName', [labels{n} ', soft']); hold on;
        semilogy(t.EbN0_dB, max(t.([names{n} '_hard']), 1e-7), [colours{n} '--' markers{n}], ...
            'LineWidth', 1, 'DisplayName', [labels{n} ', hard']);
    end
    grid on; ylim([1e-5 1]);
    xlabel('E_b/N_0 (dB)'); ylabel('BER'); title([channels{c} ' multipath']);
    legend('Location', 'southwest');
end
exportgraphics(fig, fullfile(resultsDir, sprintf('ofdm_ber_seed%d.png', seed)), 'Resolution', 150);

%% ------------------------------------------------------------------------
function selfCheck(cfg)
% Must pass before any result is trusted.
small = cfg; small.Nsym = 500;

% 1. Effectively without noise, LS at pilots equals H and every receiver decodes perfectly.
r = ofdm_link_point(small, cfg.pilotIdx, 'static', 200);
assert(r.nmseLsPilots < -150, 'LS at pilots should be exact without noise');
fields = {'berKnownHard', 'berKnownSoft', 'berLsHard', 'berLsSoft', 'berLmmseHard', 'berLmmseSoft'};
for f = fields
    assert(r.(f{1}) == 0, '%s should be zero without noise', f{1});
end

% 2. LS at pilots must match theory: NMSE = 1 / SNR, for both channels.
small.Nsym = 2000;
for ch = ["static", "fading"]
    for e = [0 10]
        r = ofdm_link_point(small, cfg.pilotIdx, char(ch), e);
        gap = abs(r.nmseLsPilots - r.expectedLsPilotsDb);
        assert(gap < 0.5, 'LS pilot NMSE off theory by %.2f dB (%s, %d dB)', gap, ch, e);
    end
end

% 3. At low SNR, LMMSE must beat LS clearly because it filters noise across pilots.
r = ofdm_link_point(small, cfg.pilotIdx, 'fading', 0);
assert(r.nmseLmmse < r.nmseLs - 3, 'LMMSE should beat LS at low SNR');

% 4. Soft decisions must beat hard decisions where errors are frequent enough to measure.
r = ofdm_link_point(small, cfg.pilotIdx, 'fading', 10);
assert(r.berKnownSoft < r.berKnownHard, 'soft decoding should beat hard decoding');
fprintf('Self-checks passed.\n');
end
