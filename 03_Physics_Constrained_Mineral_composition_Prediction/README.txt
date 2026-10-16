Module 03 - Physics-Constrained Mineral Composition Prediction
===============================================================

1. Overview
-----------
This module reproduces the synthetic-model comparison for the prediction of quartz, calcite, and clay volume fractions. Three elastic-impedance (EI) attributes at incidence angles of 8, 16, and 24 degrees are used, together with reference Poisson impedance (PI) and porosity in the GRU models. The module includes three methods:

1. Conventional statistical Bayesian inversion: estimates mineral contents from three EI attributes using a calibrated statistical forward relationship, Gaussian-mixture modeling, and Monte Carlo simulation.
2. Data-driven GRU (No RPC): predicts the three mineral fractions from EI1, EI2, EI3, reference PI, and reference porosity without an explicit rock-physics loss.
3. Rock-physics-constrained GRU (With RPC): uses the same basic GRU architecture and adds an elastic-impedance reflectivity-consistency loss derived from a differentiable rock-physics forward model.

The supplied comparison program displays the predictions of all three methods and calculates their errors. This module corresponds to the second prediction task in Stage I of the staged framework. Reference PI and porosity are used instead of outputs from Module 02 to evaluate mineral prediction without upstream prediction-error propagation.


2. Files
-----------
Modeltest.mat
  Example dataset containing the MATLAB variable Modeltest with 163 rows and 19 columns.

Minerals_statistical_Bayesian_inversion.m
  Conventional statistical Bayesian inversion. Exports Mineral_Bayesian_results.csv.

Minerals_prediction_GRU_DD.py
  Data-driven GRU baseline without RPC. Exports prediction_data_driven.csv.

Minerals_prediction_GRU_RPC.py
  Rock-physics-constrained GRU. Exports prediction_physics_constrained.csv and a model checkpoint.

well_compare_view.py
  Compares three mineral-prediction results, generates depth profiles and error plots, and prints evaluation statistics.

3. Input data
-----------
The example MATLAB file contains a numeric variable named Modeltest. The following MATLAB column numbers are used by the mineral-prediction scripts:

Column 1:  Depth, in m
Column 2:  Pe, effective pressure in MPa
Column 3:  VQUA, reference quartz volume fraction
Column 4:  Vclay, reference clay volume fraction
Column 5:  VCLC, reference calcite volume fraction
Column 9:  EI1, elastic impedance at 8 degrees
Column 10: EI2, elastic impedance at 16 degrees
Column 11: EI3, elastic impedance at 24 degrees
Column 12: PI, reference Poisson impedance
Column 13: Porosity, reference porosity fraction
Column 14: Sw, water saturation
Column 15: Density, in kg/m3
Column 16: Vp, reference P-wave velocity
Column 17: Vs, reference S-wave velocity
Column 18: fsoft, compliant-pore fraction
Column 19: alpha, pore aspect ratio

Python uses zero-based column indices. Mineral fractions and porosity are provided as decimal fractions; the comparison program converts mineral values to percentages for display. In the RPC implementation, Pe is interpreted as MPa and converted to Pa.

GRU inputs: EI1, EI2, EI3, reference PI, reference porosity.
GRU outputs: quartz, calcite, clay fractions.
Bayesian inversion inputs: EI1, EI2, EI3, with reference mineral curves used for model calibration.

The data file Modeltest.mat is a separately supplied 19-column matrix. Module 01 does not automatically export this MAT file from its Calculated_parameters.txt output.

4. Software requirements
-----------
Python 3 with numpy, scipy, pandas, matplotlib, scikit-learn, and a compatible TensorFlow/Keras installation.
MATLAB with the Statistics and Machine Learning Toolbox for fitgmdist and statset in the Bayesian inversion.
Modeltest.mat must be available in the working directory used to run the programs.

Install the required Python packages using:

python -m pip install numpy scipy pandas matplotlib scikit-learn tensorflow

The supplied Python scripts do not pin package versions, hardware settings, or global random seeds. Results may vary between installations and runs.

5. Execution procedure
-----------
Run the following programs from the Module 03 directory. Keep their generated CSV files together so that the comparison script can read them.

Step 1 - Run conventional statistical Bayesian inversion

In MATLAB, execute:

Minerals_statistical_Bayesian_inversion

Expected output:
Mineral_Bayesian_results.csv

This script fits a three-component Gaussian-mixture prior, uses 8,000 Monte Carlo samples, and computes marginal maximum-a-posteriori estimates of quartz, calcite, and clay contents. The MATLAB random seed is 2026.

Step 2 - Run the data-driven GRU baseline (No RPC)

python Minerals_prediction_GRU_DD.py

Expected output:
prediction_data_driven.csv

Step 3 - Run the rock-physics-constrained GRU (With RPC)

python Minerals_prediction_GRU_RPC.py

Expected outputs:
prediction_physics_constrained.csv
best_GRU_v8_onestage.keras

The checkpoint is saved when training loss improves. The exported CSV is generated using the in-memory model weights at the end of training; the saved checkpoint is not explicitly reloaded.

Step 4 - Compare the three methods

python "well_compare_view.py"

The comparison program requires all three CSV files in the same directory. The quoted filename is necessary in shells that interpret parentheses. It produces six TIFF figures:

Vquar_Vclc_Vclay_duibi_a.tiff
Vquar_Vclc_Vclay_duibi_b.tiff
Vquar_Vclc_Vclay_duibi_c.tiff
Vquar_Error_Comparison.tiff
Vcalcite_Error_Comparison.tiff
Vclay_Error_Comparison.tiff

The script also prints MAE, MSE, RMSE, R2, Pearson correlation, Spearman correlation, mean error, and method rankings. Its rows are compared by position, not matched by depth; the three CSV files must have the same sample ordering.

6. GRU architecture and training configuration
-----------
Both GRU scripts use the following basic architecture:

Input: EI1, EI2, EI3, PI, porosity
Dense layer: 128 units, ReLU activation and L2 regularization
GRU layer: 128 units, dropout 0.2
Output layer: 3 units, sigmoid activation
Output: normalized quartz, calcite, and clay fractions

The input shape is (number_of_samples, 5, 1). The five GRU sequence positions represent input attributes, not a continuous depth window. Both scripts use MinMaxScaler on the complete input and target dataset before prediction or training.

Configuration                         No RPC GRU                  With RPC GRU
Optimizer                             RMSprop                     RMSprop
Maximum learning rate                 0.003                       0.003
Minimum learning rate                 1e-5                        1e-5
Warm-up steps                         100                         100
Cosine period parameter               2000 steps                  2000 steps
Maximum training duration             1000 epochs                 1000 epochs
Gradient clipping                     clipnorm=1.0                clipnorm=1.0
Training procedure                    Keras minibatches           Manual full-batch updates
Loss                                  MSE                         Weighted data and physics losses



The physics model includes Voigt-Reuss-Hill mineral averaging, pore-geometry factors, pore-space stiffness, fluid substitution, and elastic-impedance calculations. The physical term compares adjacent-depth EI reflectivities at 8, 16, and 24 degrees. Other loss terms defined in the source code, including mineral-sum, calcite-edge, and calcite-peak losses, have zero weights in the supplied configuration.

The learning-rate scheduler is named PS_CALR in the source files. The supplied implementation uses a linear warm-up followed by a cosine function and does not explicitly implement a periodic phase shift or restart. Verify the implementation against the manuscript definition before claiming exact equivalence.

7. Interpretation and reproducibility notes
-----------
The example is designed for controlled comparison of the mineral-prediction task. Using reference PI and porosity avoids errors propagated from Module 02.

The mineral fractions in exported CSV files are stored as decimal fractions. The comparison plots display these values as percentages. Before interpreting combined error profiles, verify that the reference curves, depth ordering, and sample counts are consistent across all outputs.


8. Place in the complete repository
-----------
This is Module 03: Physics-Constrained Mineral Composition Prediction. Module 01 supplies the rock-physics background and elastic-property modeling. Module 02 evaluates PI and porosity prediction. Module 03 evaluates mineral-composition prediction from EI and reference PI/porosity. Module 04 evaluates rock-physics-guided lithology classification.

The synthetic module tests evaluate individual prediction tasks under controlled inputs. In contrast, the field-data application propagates predicted parameters through the complete staged framework and evaluates its integrated performance.
