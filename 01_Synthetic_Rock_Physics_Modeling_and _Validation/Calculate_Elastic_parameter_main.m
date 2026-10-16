%% =========================================================
% Vp_obs_Vs_obs_forward_calculate_main.m
% Calculate Vp_obs and Vs_obs using
% Rock_physics_forward_alpha_fsoft.m
% Input txt:
% depth Pe vqua clay vclc Feldspar Dolomite Kerogen
% por sw den vp VS fsoft alpha
%
% =========================================================
clear;
clc;
close all;
format long g;
%% ===================== Read data =========================
data_file = ...
    'Model_data_used_to_Rock_physics_forward.txt';
% First line is header:
% depth Pe vqua clay vclc Feldspar Dolomite Kerogen
% por sw den vp VS fsoft alpha
raw = importdata(data_file);
% numeric matrix
WELL = raw.data;
%% ===================== Extract variables =================
Depth = WELL(:,1);
Pe_in = WELL(:,2);
Con = WELL(:,3:7);
Por = WELL(:,9);
Sw = WELL(:,10);
Rho = WELL(:,11);
Vp_obs = WELL(:,12);
Vs_obs = WELL(:,13);
fsoft = WELL(:,14);
alpha = WELL(:,15);
Lith = WELL(:,16);

%% ===================== Unit conversion ===================
% According to your SA inversion main program:
%
% pressure_is_MPa = true
%
% Pe = Pe_in * 1e6
Pe = Pe_in * 1e6;
%% ===================== Mineral modulus ===================
% Same as:
% main_vp_vs_joint_inversion_alpha_fsoft.m
Modu = [

36.6 45.0;

21.0 7.0;

76.8 32.0;

37.5 15.0;

54.9 43.6;

2.9 2.7;

2.0 0;

0.7 0

]*1e9;
%% ===================== Rock physics model ===============
model.A = 0.0047;
model.B = 0.001;
model.use_alpha_PQ = true;
%% ===================== Forward calculation ===============
N = length(Vp_obs);
Vp_cal = nan(N,1);
Vs_cal = nan(N,1);
Ksat = nan(N,1);
Musat = nan(N,1);
for i = 1:N


    [Vp_cal(i),Vs_cal(i),...
     Ksat(i),Musat(i),detail] = ...
        Rock_physics_forward(...
        Modu,...
        Con(i,:),...
        Pe(i),...
        Sw(i),...
        Por(i),...
        Rho(i),...
        fsoft(i),...
        alpha(i),...
        model);


end
script_path = fileparts(mfilename('fullpath'));
outdir = fullfile(script_path, 'Figure');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

%% ================= Vp_obs ================================
figure('Name','Vp_obs prediction',...
       'Color','w');
plot(Vp_obs,Depth,...
    'k-',...
    'LineWidth',1.5);
hold on;
plot(Vp_cal,Depth,...
    'r-',...
    'LineWidth',1.5);
set(gca,'YDir','reverse');
xlim([3000 5500]);
ylim([min(Depth),max(Depth)]);
xlabel('Vp_obs/(m\cdot s^{-1})');
ylabel('Depth/m');
set(gca,...
    'FontSize',16,...
    'FontName','Times New Roman');
box on;
set(gca,...
    'TickDir','in',...
    'LineWidth',1.0);
legend({'Observed','Predicted'},...
    'Location','southwest',...
    'Box','off');
print(gcf,...
    fullfile(outdir,'Vp_obs_prediction.tif'),...
    '-dtiff',...
    '-r600');
%% ================= Vs_obs ================================
figure('Name','Vs_obs prediction',...
       'Color','w');
plot(Vs_obs,Depth,...
    'k-',...
    'LineWidth',1.5);
hold on;
plot(Vs_cal,Depth,...
    'r-',...
    'LineWidth',1.5);
set(gca,'YDir','reverse');
xlim([2000 3500]);
ylim([min(Depth),max(Depth)]);
xlabel('Vs_obs/(m\cdot s^{-1})');
ylabel('Depth/m');
set(gca,...
    'FontSize',16,...
    'FontName','Times New Roman');
box on;
set(gca,...
    'TickDir','in',...
    'LineWidth',1.0);
legend({'Observed','Predicted'},...
    'Location','southwest',...
    'Box','off');
print(gcf,...
    fullfile(outdir,'Vs_obs_prediction.tif'),...
    '-dtiff',...
    '-r600');

%% ================= Vp_obs ================================
figure('Name','Vp_obs error distribution histogram','Color','w');
vp_hist = (Vp_cal - Vp_obs) ./ Vp_obs;
vp_hist = vp_hist(...
    isfinite(vp_hist)...
    & Vp_obs > 0 ...
    & Vp_obs ~= -999.25);

histfit(vp_hist);
h = get(gca, 'Children');
if numel(h) >= 2
    set(h(1), 'Color', [195,39,43]/255);     
    set(h(2), 'FaceColor', [214,236,240]/255); 
end

xlim([-0.2 0.2]);
ylim([1 250]);
box on;
xlabel('P-wave velocity prediction error');
ylabel('Number of samples');
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');

vp_save_path = fullfile(outdir, 'Vp_obs_error_distribution_histogram_forward.tif');
print(gcf, vp_save_path, '-dtiff', '-r600');

%% ================= Vs_obs ================================
figure('Name','Vs_obs error distribution histogram','Color','w');
vs_hist = (Vs_cal - Vs_obs) ./ Vs_obs;
vs_hist = vs_hist(...
    isfinite(vs_hist)...
    & Vs_obs > 0 ...
    & Vs_obs ~= -999.25);

histfit(vs_hist);
h = get(gca, 'Children');
if numel(h) >= 2
    set(h(1), 'Color', [195,39,43]/255);
    set(h(2), 'FaceColor', [214,236,240]/255, 'EdgeColor', 'k');
end

xlim([-0.2 0.2]);
ylim([1 250]);
box on;
xlabel('S-wave velocity prediction error');
ylabel('Number of samples');
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');

vs_save_path = fullfile(outdir, 'Vs_obs_error_distribution_histogram_forward.tif');
print(gcf, vs_save_path, '-dtiff', '-r600');

%% =========================================================
% Velocity prediction error statistics
% =========================================================


fprintf('\n========================================\n');
fprintf(' Rock physics forward prediction error\n');
fprintf('========================================\n');


%% ================= Vp_obs error ==============================
valid_vp = isfinite(Vp_cal) ...
    & isfinite(Vp_obs) ...
    & Vp_obs > 0 ...
    & Vp_obs ~= -999.25;
vp_relative_error = ...
    (Vp_cal(valid_vp)-Vp_obs(valid_vp)) ...
    ./ Vp_obs(valid_vp);
vp_abs_error = abs(vp_relative_error);
Vp_obs_MARE = 100*mean(vp_abs_error);
Vp_obs_ME = 100*mean(vp_relative_error);
Vp_obs_RMSE = sqrt( ...
    mean((Vp_cal(valid_vp)-Vp_obs(valid_vp)).^2) );
Vp_obs_MAX = max(abs(vp_relative_error))*100;
fprintf('\n------------- Vp_obs prediction -------------\n');
fprintf('Valid samples       = %d\n',...
    sum(valid_vp));
fprintf('MARE                = %.4f %%\n',...
    Vp_obs_MARE);
fprintf('Mean relative error = %.4f %%\n',...
    Vp_obs_ME);
fprintf('RMSE                = %.3f m/s\n',...
    Vp_obs_RMSE);
fprintf('Maximum error       = %.4f %%\n',...
    Vp_obs_MAX);
%% ================= Vs_obs error ==============================
valid_vs = isfinite(Vs_cal) ...
    & isfinite(Vs_obs) ...
    & Vs_obs > 0 ...
    & Vs_obs ~= -999.25;
vs_relative_error = ...
    (Vs_cal(valid_vs)-Vs_obs(valid_vs)) ...
    ./ Vs_obs(valid_vs);
vs_abs_error = abs(vs_relative_error);
Vs_obs_MARE = 100*mean(vs_abs_error);
Vs_obs_ME = 100*mean(vs_relative_error);
Vs_obs_RMSE = sqrt( ...
    mean((Vs_cal(valid_vs)-Vs_obs(valid_vs)).^2) );
Vs_obs_MAX = max(abs(vs_relative_error))*100;
fprintf('\n------------- Vs_obs prediction -------------\n');
fprintf('Valid samples       = %d\n',...
    sum(valid_vs));
fprintf('MARE                = %.4f %%\n',...
    Vs_obs_MARE);
fprintf('Mean relative error = %.4f %%\n',...
    Vs_obs_ME);
fprintf('RMSE                = %.3f m/s\n',...
    Vs_obs_RMSE);
fprintf('Maximum error       = %.4f %%\n',...
    Vs_obs_MAX);
fprintf('\n========================================\n');
%% Calculate mean values using valid samples only
if length(Vp_obs) ~= length(Vs_obs) || length(Vp_obs) ~= length(Rho)
    error('Vp_obs, Vs_obs, and Density must have the same number of samples.');
end

valid = isfinite(Vp_obs) & isfinite(Vs_obs) & isfinite(Rho) & ...
        Vp_obs > 0 & Vs_obs > 0 & Rho > 0;

if ~any(valid)
    error('No valid Vp_obs, Vs_obs, and Density samples were found.');
end


mean_Vp_obs      = mean(Vp_obs(valid));
mean_Vs_obs      = mean(Vs_obs(valid));
mean_Rho = mean(Rho(valid));

fprintf('Mean Vp_obs      = %.6f\n', mean_Vp_obs);
fprintf('Mean Vs_obs      = %.6f\n', mean_Vs_obs);
fprintf('Mean Density = %.6f\n', mean_Rho);

%% ==================== Calculate EI at Three Angles ======================
EI1 = nan(size(Vp_obs));
EI2 = nan(size(Vp_obs));
EI3 = nan(size(Vp_obs));
% Three incidence angles in degrees
theta_deg = [8, 16, 24];

% ---------- EI1 ----------
theta1 = deg2rad(theta_deg(1));

EI1(valid) = ...
    (mean_Vp_obs * mean_Rho) .* ...
    (Vp_obs(valid) ./ mean_Vp_obs) .^ (1 + tan(theta1).^2) .* ...
    (Vs_obs(valid) ./ mean_Vs_obs) .^ (-8 * 0.25 * sin(theta1).^2) .* ...
    (Rho(valid) ./ mean_Rho) .^ (1 - 4 * 0.25 * sin(theta1).^2);

% ---------- EI2 ----------
theta2 = deg2rad(theta_deg(2));

EI2(valid) = ...
    (mean_Vp_obs * mean_Rho) .* ...
    (Vp_obs(valid) ./ mean_Vp_obs) .^ (1 + tan(theta2).^2) .* ...
    (Vs_obs(valid) ./ mean_Vs_obs) .^ (-8 * 0.25 * sin(theta2).^2) .* ...
    (Rho(valid) ./ mean_Rho) .^ (1 - 4 * 0.25 * sin(theta2).^2);

% ---------- EI3 ----------
theta3 = deg2rad(theta_deg(3));

EI3(valid) = ...
    (mean_Vp_obs * mean_Rho) .* ...
    (Vp_obs(valid) ./ mean_Vp_obs) .^ (1 + tan(theta3).^2) .* ...
    (Vs_obs(valid) ./ mean_Vs_obs) .^ (-8 * 0.25 * sin(theta3).^2) .* ...
    (Rho(valid) ./ mean_Rho) .^ (1 - 4 * 0.25 * sin(theta3).^2);
%% ==================== Calculate Poisson Impedance ====================

% Poisson's ratio calculated from P-wave and S-wave velocities:
% nu = (Vp_obs^2 - 2*Vs_obs^2) / (2*(Vp_obs^2 - Vs_obs^2))
%
% Poisson impedance:
% PI = nu * Density

poisson_ratio = ...
    (Vp_obs.^2 - 2 .* Vs_obs.^2) ./ ...
    (2 .* (Vp_obs.^2 - Vs_obs.^2));

PI = poisson_ratio .* Rho;
%% ======================== Plot Results =========================
figure('Color', 'w');

if isempty(Depth)
    xaxis_data = (1:length(vp))';
    yLabelText = 'Sample';
else
    xaxis_data = Depth;
    yLabelText = 'Depth';
end

subplot(1,4,1);
plot(EI1, xaxis_data, 'k-', 'LineWidth', 1.0);
set(gca, 'YDir', 'reverse');
xlabel(sprintf('EI1 (%g^\\circ)', theta_deg(1)));
ylabel(yLabelText);
grid on;

subplot(1,4,2);
plot(EI2, xaxis_data, 'k-', 'LineWidth', 1.0);
set(gca, 'YDir', 'reverse');
xlabel(sprintf('EI2 (%g^\\circ)', theta_deg(2)));
ylabel(yLabelText);
grid on;

subplot(1,4,3);
plot(EI3, xaxis_data, 'k-', 'LineWidth', 1.0);
set(gca, 'YDir', 'reverse');
xlabel(sprintf('EI3 (%g^\\circ)', theta_deg(3)));
ylabel(yLabelText);
grid on;

subplot(1,4,4);
plot(PI, xaxis_data, 'k-', 'LineWidth', 1.0);
set(gca, 'YDir', 'reverse');
xlabel(sprintf('PI'));
ylabel(yLabelText);
grid on;

sgtitle('Elastic Impedance at Three Angles');
%%
savefile = fullfile('Calculated_parameters.txt');
out_data = [Depth(:), Vp_obs(:), Vp_cal(:), Vs_obs(:), Vs_cal(:), EI1(:), EI2(:), EI3(:),...
    Con(:,1),Con(:,2),Con(:,3), PI(:), Por(:), Lith(:), Rho(:)];

fid = fopen(savefile,'w');
fprintf(fid,'Depth Vp_obs Vp_cal Vs_obs Vs_cal EI1 EI2 EI3 VQUA Vclay VCLC PI Por Lith RHOB\n');
for i = 1:size(out_data,1)
    fprintf(fid,'%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n', out_data(i,1),out_data(i,2),out_data(i,3),out_data(i,4),out_data(i,5),out_data(i,6),...
        out_data(i,7),out_data(i,8),out_data(i,9),out_data(i,10),out_data(i,11),out_data(i,12),out_data(i,13),out_data(i,14),out_data(i,15));
end
fclose(fid);