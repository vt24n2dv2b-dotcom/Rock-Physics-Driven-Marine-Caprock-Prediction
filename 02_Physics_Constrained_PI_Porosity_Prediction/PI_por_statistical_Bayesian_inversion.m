%% Statistical Bayesian inversion of Poisson impedance and porosity from EI1/EI2
% Outputs: Poisson impedance (PI) and porosity (por).
% Method: joint Gaussian-mixture prior, regularized statistical forward
% model, Monte Carlo simulation, and conditional Gaussian-mixture inversion.
% This is statistical Bayesian inversion, NOT neural-network/deep learning.
% Requires Statistics and Machine Learning Toolbox (fitgmdist, statset).
% Place Modeltest.mat in the MATLAB current working directory.
% Reference PI/por curves are used for statistical calibration and in-well comparison.
% Only EI1 and EI2 are required during inversion.

clc; clear; close all;
rng(2026, 'twister');

%% 1. Load observed impedances and reference PI/porosity curves
D = load('Modeltest.mat');
assert(isfield(D,'Modeltest'), ...
    'The MAT file must contain Modeltest.');
well = D.Modeltest;
assert(isnumeric(well) && ismatrix(well) && size(well,2)>=13, ...
    'Modeltest must have at least 13 numeric columns.');

z   = well(:,1);                     % Depth
EI1 = well(:,9);                     % Elastic impedance 1
EI2 = well(:,10);                    % Elastic impedance 2
PI  = well(:,12);                    % Poisson impedance (NOT pi = 3.14159)
por = well(:,13);                    % Porosity

observedEI = [EI1, EI2];
referenceY = [PI, por];              % Original reference measurement units
propertyNames = {'Poisson impedance (PI)','Porosity (por)'};
numRows = size(well,1);
numEI = 2;
numProperties = 2;

% Calibration rows contain reference PI/por AND EI1/EI2 observations.
% Inversion rows need depth and EI1/EI2 observations only.
% These logical indices do NOT represent deep-learning data partitions.
calibrationRows = all(isfinite([z, observedEI, referenceY]),2);
inversionRows = all(isfinite([z, observedEI]),2);
assert(nnz(calibrationRows)>=30, ...
    'At least 30 complete calibration rows are required.');
assert(any(inversionRows),'No valid rows are available for inversion.');

%% 2. Express porosity internally as a fraction
% PI retains its original units, since it is not a fractional property.
% Porosity is converted from percent to fraction only if needed.
% Set porScale manually when the original unit is ambiguous.
porScale = 1;
if max(referenceY(calibrationRows,2))>1.5
    porScale = 100;
end
Ycal = referenceY(calibrationRows,:);
Ycal(:,2) = Ycal(:,2)/porScale;
assert(all(Ycal(:,2)>=-1e-6 & Ycal(:,2)<=1+1e-6), ...
    'Check units of the porosity reference column.');
Ycal(:,2) = min(max(Ycal(:,2),0),1);
Xcal = observedEI(calibrationRows,:);
Xinv = observedEI(inversionRows,:);
zInv = z(inversionRows);
fprintf('Calibration rows: %d; inversion rows: %d.\n', ...
    nnz(calibrationRows),nnz(inversionRows));
fprintf('Porosity measurement scale: %g.\n',porScale);

%% 3. Standardize the calibration data
% Standardization does not change the final measurement units.
muY = mean(Ycal,1);
sdY = std(Ycal,0,1);
sdY(sdY<1e-8) = 1;
YcalZ = bsxfun(@rdivide,bsxfun(@minus,Ycal,muY),sdY);
muX = mean(Xcal,1);
sdX = std(Xcal,0,1);
sdX(sdX<1e-8) = 1;
XcalZ = bsxfun(@rdivide,bsxfun(@minus,Xcal,muX),sdX);

%% 4. Establish a deterministic statistical rock-physics forward relation
% This is a calibrated surrogate model: observed EI = f(properties) + error.
% A small ridge penalty helps stabilize correlated reference properties.
A = [ones(size(YcalZ,1),1),YcalZ];
regularization = 1e-2;
penalty = diag([0,ones(1,numProperties)]);  % Do not penalize intercept
forwardCoefficients = ...
    (A'*A+regularization*penalty)\(A'*XcalZ);
XfittedZ = A*forwardCoefficients;
residualStdZ = max(std(XcalZ-XfittedZ,0,1),0.03);

%% 5. Construct a joint Gaussian-mixture prior and draw Monte Carlo samples
% A joint prior preserves covariance among the target properties.
numComponents = 3;
numMonteCarlo = 10000;
fitOptions = statset('MaxIter',300,'Display','off');
priorGMM = fitgmdist(YcalZ,numComponents, ...
    'CovarianceType','full','RegularizationValue',1e-2, ...
    'Replicates',2,'Options',fitOptions);
priorDrawZ = random(priorGMM,numMonteCarlo);
propertyDraw = bsxfun(@plus,bsxfun(@times,priorDrawZ,sdY),muY);

%% 6. Enforce porosity bounds in the simulated prior
% PI can be positive or negative depending on the impedance definition.
% Do not apply a 0-1 bound or force PI to be positive.
propertyDraw(:,2) = min(max(propertyDraw(:,2),0),1);

%% 7. Generate synthetic EI observations and fit their joint GMM
% Monte Carlo EI values are computed from the forward relation with noise.
propertyDrawZ = bsxfun(@rdivide, ...
    bsxfun(@minus,propertyDraw,muY),sdY);
syntheticEIZ = [ones(numMonteCarlo,1),propertyDrawZ]* ...
    forwardCoefficients + ...
    bsxfun(@times,randn(numMonteCarlo,numEI),residualStdZ);

% Joint variable order: all observed EI values, then target properties.
jointDrawZ = [syntheticEIZ,propertyDrawZ];
jointGMM = fitgmdist(jointDrawZ,numComponents, ...
    'CovarianceType','full','RegularizationValue',1e-2, ...
    'Replicates',2,'Options',fitOptions);

%% 8. Calculate the conditional Gaussian mixture P(properties | EI)
% For each Gaussian component, partition the covariance matrix into
% observed impedance (X) and target properties (Y):
%   mu(Y|X) = muY + SigmaYX * inv(SigmaXX) * (X-muX)
%   S(Y|X)  = SYY - SigmaYX * inv(SigmaXX) * SigmaXY
% Use linear solves rather than explicitly computing an inverse.
XinvZ = bsxfun(@rdivide,bsxfun(@minus,Xinv,muX),sdX);
numInversionRows = size(XinvZ,1);
K = jointGMM.NumComponents;
conditionalMeanZ = zeros(numInversionRows,numProperties,K);
conditionalVarZ  = zeros(K,numProperties);
logComponentWeight = zeros(numInversionRows,K);

for k = 1:K
    componentMean = jointGMM.mu(k,:);
    componentCov  = jointGMM.Sigma(:,:,k);
    idxX = 1:numEI;
    idxY = numEI+(1:numProperties);
    Sxx = componentCov(idxX,idxX);
    Sxy = componentCov(idxX,idxY);
    Syx = componentCov(idxY,idxX);
    Syy = componentCov(idxY,idxY);
    Sxx = (Sxx+Sxx')/2 + 1e-10*eye(numEI);

    centeredX = bsxfun(@minus,XinvZ,componentMean(idxX));
    conditionalMeanZ(:,:,k) = bsxfun(@plus, ...
        centeredX*(Sxx\Sxy),componentMean(idxY));
    condCov = Syy - Syx*(Sxx\Sxy);
    conditionalVarZ(k,:) = max(diag((condCov+condCov')/2)',1e-6);

    % Evaluate log P(EI|component) to update GMM component probabilities.
    R = chol(Sxx);
    scaledX = centeredX/R;
    squaredDistance = sum(scaledX.^2,2);
    logDetSxx = 2*sum(log(diag(R)));
    logComponentWeight(:,k) = ...
        log(max(jointGMM.ComponentProportion(k),realmin)) ...
        -0.5*(numEI*log(2*pi)+logDetSxx+squaredDistance);
end

% Row-wise log-sum-exp normalization avoids numerical underflow.
largestLogWeight = max(logComponentWeight,[],2);
componentWeight = exp(bsxfun(@minus,logComponentWeight,largestLogWeight));
componentWeight = bsxfun(@rdivide,componentWeight, ...
    sum(componentWeight,2));

%% 9. Compute marginal posterior probability density and MAP estimates
% PI uses its original units and a range slightly wider than its references.
% Porosity is mapped to the physically meaningful 0-1 interval.
numGrid = 501;
propertyGrid = cell(1,numProperties);
piObserved = Ycal(:,1);
piSpan = max(piObserved)-min(piObserved);
assert(piSpan>1e-10,'The reference PI curve must have variation.');
piMargin = 0.15*piSpan;
propertyGrid{1} = linspace(min(piObserved)-piMargin, ...
    max(piObserved)+piMargin,numGrid);
propertyGrid{2} = linspace(0,1,numGrid);
posteriorPDF = cell(1,numProperties);
mapEstimate = nan(numInversionRows,numProperties);
posteriorMean = zeros(numInversionRows,numProperties);

for j = 1:numProperties
    gridValues = propertyGrid{j};
    gridDensity = zeros(numInversionRows,numGrid);
    for k = 1:K
        meanValue = conditionalMeanZ(:,j,k)*sdY(j)+muY(j);
        sigmaValue = max(sqrt(conditionalVarZ(k,j))*sdY(j), ...
            max(0.005*sdY(j),1e-8));
        deviation = bsxfun(@rdivide, ...
            bsxfun(@minus,gridValues,meanValue),sigmaValue);
        gaussianPDF = exp(-0.5*deviation.^2)/ ...
            (sqrt(2*pi)*sigmaValue);
        gridDensity = gridDensity + bsxfun(@times, ...
            componentWeight(:,k),gaussianPDF);
        posteriorMean(:,j) = posteriorMean(:,j)+ ...
            componentWeight(:,k).*meanValue;
    end

    % Normalize each depth-specific density over its allowed grid.
    integralOverGrid = trapz(gridValues,gridDensity,2);
    invalidDensity = ~isfinite(integralOverGrid) | ...
        integralOverGrid<=realmin;
    if any(invalidDensity)
        fallbackMean = min(max(posteriorMean(invalidDensity,j), ...
            gridValues(1)),gridValues(end));
        fallbackSigma = max(0.02*(gridValues(end)-gridValues(1)),1e-8);
        delta = bsxfun(@rdivide, ...
            bsxfun(@minus,gridValues,fallbackMean),fallbackSigma);
        gridDensity(invalidDensity,:) = exp(-0.5*delta.^2);
        integralOverGrid = trapz(gridValues,gridDensity,2);
    end
    posteriorPDF{j} = bsxfun(@rdivide,gridDensity, ...
        max(integralOverGrid,realmin));
    [~,bestIndex] = max(posteriorPDF{j},[],2);
    mapEstimate(:,j) = gridValues(bestIndex);
end

%% 10. Enforce the porosity bound and restore original measurement units
% Do not force PI to be positive; only porosity is restricted to [0,1].
mapEstimate(:,2) = min(max(mapEstimate(:,2),0),0.3);
propertyEstimate = mapEstimate;
propertyEstimate(:,2) = mapEstimate(:,2)*porScale;

PI_pre = nan(numRows,1);%创建完整井深长度的预测曲线
por_pre = nan(numRows,1);%创建完整井深长度的预测曲线
PI_pre(inversionRows)  = propertyEstimate(:,1);
por_pre(inversionRows) = propertyEstimate(:,2);
predictionCurves = [PI_pre,por_pre];
propertyGrid{2} = propertyGrid{2}*porScale;

%% 11. Quantify agreement with the reference curves (same well)
% These RMSE/MAE/R2 values are calibration-well agreement only; do not
% report them as predictive skill on a separate blind well.
referenceAtInversion = referenceY(inversionRows,:);
for j = 1:numProperties
    comparableRows = isfinite(referenceAtInversion(:,j));
    if nnz(comparableRows)<2
        continue;
    end
    measured = referenceAtInversion(comparableRows,j);
    estimated = propertyEstimate(comparableRows,j);
    rmse = sqrt(mean((measured-estimated).^2));
    mae = mean(abs(measured-estimated));
    referenceVariation = sum((measured-mean(measured)).^2);
    if referenceVariation>eps
        rSquared = 1-sum((measured-estimated).^2)/referenceVariation;
    else
        rSquared = NaN;
    end
    fprintf('%s: RMSE=%.5g, MAE=%.5g, R2=%.4f (same-well fit)\n', ...
        propertyNames{j},rmse,mae,rSquared);
end

%% 12. Plot impedance inputs, inversion curves, and posterior densities
inputLabels = {'EI1','EI2'};
figure('Name','Observed EI1 and EI2','Color','w');
for j = 1:numEI
    subplot(1,numEI,j);
    plot(observedEI(:,j),z,'k-','LineWidth',1);
    set(gca,'YDir','reverse'); grid on;
    xlabel(inputLabels{j});
    if j==1, ylabel('Depth (m)'); end
end

figure('Name','Poisson impedance and porosity inversion', ...
    'Color','w');
for j = 1:numProperties
    subplot(1,numProperties,j);
    plot(referenceY(:,j),z,'b-','LineWidth',1.1); hold on;
    plot(predictionCurves(:,j),z,'r--','LineWidth',1.1);
    set(gca,'YDir','reverse'); grid on;
    xlabel(propertyNames{j});
    if j==1, ylabel('Depth (m)'); end
    legend({'Reference','Inverted'},'Location','best');
end

[zSorted,order] = sort(zInv);
figure('Name','PI and porosity posterior densities', ...
    'Color','w');
for j = 1:numProperties
    subplot(1,numProperties,j);
    [GX,GZ] = meshgrid(propertyGrid{j},zSorted);
    p = posteriorPDF{j};
    surf(GX,GZ,zeros(size(GX)),p(order,:), ...
        'EdgeColor','none');
    view(2); set(gca,'YDir','reverse');
    xlabel(propertyNames{j});
    if j==1, ylabel('Depth (m)'); end
    colorbar;
end

%% 13. Save results with reference and inverted values
result = [z, PI, PI_pre, por, por_pre];
outputCSV = 'PI_por_Bayesian_results.csv';
fid = fopen(outputCSV,'w');
assert(fid>=0,'Unable to create the output CSV file.');
% Write column headers
fprintf(fid, ...
    'Depth,True PI,Pred PI,True Por,Pred Por\n');
% Write numerical results
fprintf(fid, '%.10g,%.10g,%.10g,%.10g,%.10g\n', result.');
% Close the file
fclose(fid);