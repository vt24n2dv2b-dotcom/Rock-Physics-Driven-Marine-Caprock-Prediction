%% Lithology-specific validation of rock-physics modeling
% Input:
% Depth Vp_obs Vp_cal Vs_obs Vs_cal EI1 EI2 EI3
% VQUA Vclay VCLC PI por Lith RHOB
clear;
clc;
close all;
format long g;

%% ===================== Read data =========================
data_file = 'Calculated_parameters.txt';
if ~isfile(data_file)
    error('Cannot find file: %s',data_file);
end
raw = importdata(data_file);
WELL = raw.data;
fprintf('Total samples: %d\n',size(WELL,1));

%% ===================== Extract variables =================
Depth = WELL(:,1);
Vp_obs = WELL(:,2);
Vp_cal = WELL(:,3);
Vs_obs = WELL(:,4);
Vs_cal = WELL(:,5);
EI1 = WELL(:,6);
EI2 = WELL(:,7);
EI3 = WELL(:,8);
VQUA  = WELL(:,9);
Vclay = WELL(:,10);
VCLC  = WELL(:,11);
PI  = WELL(:,12);
por = WELL(:,13);
Lith = WELL(:,14);
RHOB = WELL(:,15);

%% ===================== Lithology definitions =============
lithCodes = [1 2 3];
lithNames = {
    'Sandstone'
    'Mudstone'
    'Calcareous sandstone'
};
nLith = length(lithCodes);

lithColors = [
    1.0000  0.8000  0.0000; 
    0.0000  0.0000  0.0000;  
    0.0000  0  1   
];

%% ===================== Output folder ======================
outDir='Figure';
if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% ===================== Vp crossplot ======================
figure;
hold on;
box on;

for k=1:nLith
    idx = Lith == lithCodes(k);
    scatter(Vp_obs(idx),...
        Vp_cal(idx),...
        35,...
        lithColors(k,:),...       % 按岩性指定颜色
        'filled',...
        'MarkerEdgeColor', 'none',...  % 去掉散点外边框，更干净
        'DisplayName', lithNames{k});
end

axis_range = [4500 5200];
plot(axis_range, axis_range, 'k--', 'DisplayName', 'y=x');
xlim(axis_range);
ylim(axis_range);

xlabel('Observed Vp (m/s)');
ylabel('Predicted Vp (m/s)');
legend('FontSize', 16, 'Location', 'northwest', 'Box', 'off');

set(gca,...
    'FontSize',16,...
    'FontName','Times New Roman');

print(gcf,...
    fullfile(outDir, 'Vp_prediction_crossplot.tif'),...
    '-dtiff',...
    '-r600');

%% ===================== Vs crossplot ======================
figure;
hold on;
box on;

for k=1:nLith
    idx = Lith == lithCodes(k);
    scatter(Vs_obs(idx),...
        Vs_cal(idx),...
        35,...
        lithColors(k,:),...       % 复用同一套配色
        'filled',...
        'MarkerEdgeColor', 'none',...
        'DisplayName', lithNames{k});
end

axis_range = [2700 3200];
plot(axis_range, axis_range, 'k--', 'DisplayName', 'y=x');
xlim(axis_range);
ylim(axis_range);

xlabel('Observed Vs (m/s)');
ylabel('Predicted Vs (m/s)');
legend('FontSize', 16, 'Location', 'northwest', 'Box', 'off');

set(gca,...
    'FontSize',16,...
    'FontName','Times New Roman');

print(gcf,...
    fullfile(outDir, 'Vs_prediction_crossplot.tif'),...
    '-dtiff',...
    '-r600');