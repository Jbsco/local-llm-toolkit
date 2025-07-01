% MATLAB script to plot statistics from CSV
close all; clear all; clc;

% Assumes the CSV file is in the current folder

% Read the table
filename = 'benchmark_32B_results_28:2:44_bench_models_17:57:10.csv';
opts = detectImportOptions(filename);
opts = setvartype(opts, 'model', 'string');  % Ensure model is read as string
data = readtable(filename, opts);

% Calculate memory delta
data.vram_diff_MB = data.vram_after_MB - data.vram_before_MB;

% Unique models
models = unique(data.model);
colors = lines(length(models));  % Distinct colors

% Plot tokens per second vs gpu_layers
figure('Name', 'Tokens/sec vs GPU Layers');
hold on;
for i = 1:length(models)
    model_data = data(data.model == models(i), :);
    plot(model_data.gpu_layers, model_data.tokens_per_sec, '-o', ...
        'DisplayName', models(i), 'Color', colors(i,:));

    x = model_data.gpu_layers;
    y = model_data.tokens_per_sec;

    model_label = char(models(i));  % Convert string to char vector
    text(x(1), y(1), model_label, 'Color', colors(i,:), ...
        'FontSize', 10, 'Interpreter', 'none', 'HorizontalAlignment', 'left');
end
xlabel('GPU Layers');
ylabel('Tokens per Second');
title('Throughput vs GPU Layers');
grid on;
hold off;

% Plot total_ms vs gpu_layers
figure('Name', 'Total Time (ms) vs GPU Layers');
hold on;
for i = 1:length(models)
    model_data = data(data.model == models(i), :);
    plot(model_data.gpu_layers, model_data.total_ms, '-o', ...
        'DisplayName', models(i), 'Color', colors(i,:));

    x = model_data.gpu_layers;
    y = model_data.total_ms;

    model_label = char(models(i));  % Convert string to char vector
    text(x(1), y(1), model_label, 'Color', colors(i,:), ...
        'FontSize', 10, 'Interpreter', 'none', 'HorizontalAlignment', 'left');
end
xlabel('GPU Layers');
ylabel('Total Time (ms)');
title('Total Evaluation Time vs GPU Layers');
grid on;
hold off;

% Plot: VRAM Difference (After - Before) vs GPU Layers (all models)
figure('Name', 'VRAM Delta vs GPU Layers (All Models)');
hold on;
for i = 1:length(models)
    model_data = data(data.model == models(i), :);
    plot(model_data.gpu_layers, model_data.vram_diff_MB, '-o', ...
        'DisplayName', models(i), 'Color', colors(i,:));

    x = model_data.gpu_layers;
    y = model_data.vram_diff_MB;

    model_label = char(models(i));  % Convert string to char vector
    text(x(1), y(1), model_label, 'Color', colors(i,:), ...
        'FontSize', 10, 'Interpreter', 'none', 'HorizontalAlignment', 'left');
end
xlabel('GPU Layers');
ylabel('VRAM Delta (MB)');
title('VRAM Usage Change (After - Before) vs GPU Layers');
grid on;
hold off;

