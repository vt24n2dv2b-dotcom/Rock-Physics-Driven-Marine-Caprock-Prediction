Module 02 — Physics-Constrained PI–Porosity Prediction
========================================================
1. Overview
-----------
This module reproduces the synthetic-model comparison for "Poisson impedance (PI)" and "porosity" prediction. Two elastic-impedance (EI) attributes at incidence angles of 8° and 16° are used as inputs. The module includes three methods:

1. **Conventional statistical Bayesian inversion**: estimates PI and porosity using a calibrated statistical forward relationship, Gaussian-mixture modeling, and Monte Carlo simulation.
2. **Data-driven GRU (No RPC)**: predicts PI and porosity from the two EI inputs using supervised learning, without an explicit rock-physics loss.
3. **Rock-physics-constrained GRU (With RPC)**: uses the same basic GRU architecture, with an additional reflectivity-consistency loss derived from the PI–porosity AVO approximation.

The three methods can be compared using the supplied plotting and evaluation script. This module corresponds to the **first prediction task** of the staged framework. It is designed to evaluate this task without propagating predictions from any upstream learned module.

**Interpretation of results:** As supplied, the GRU scripts fit the complete example dataset and report errors on those same samples; the Bayesian script calibrates its statistical model using available reference curves from the same dataset. Thus, the reported agreement is **same-dataset fitting/calibration performance, not a held-out blind-test result**. Independent field blind-well validation is a separate experiment in the manuscript.

2. Files
-----------
| File | Purpose |
| --- | --- |
| Modeltest.mat | Example data file; contains a MATLAB variable named "Modeltest" with "163 rows × 19 columns". |
| PI_por_statistical_Bayesian_inversion.m | Conventional statistical Bayesian baseline; exports "PI_por_Bayesian_results.csv". |
| PI_Por_prediction_GRU_DD.py.py | Purely data-driven GRU baseline (No RPC); exports "prediction_data_driven.csv". |
| PI_Por_prediction_GRU_RPC.py | Rock-physics-constrained GRU (With RPC); exports "prediction_physics_constrained.csv". |
| well_compare_view.py | Plots the three methods together, generates error profiles, and prints comparison statistics. |


3. Input data
-----------
The provided MATLAB file contains the numeric variable "Modeltest" with 163 samples and 19 columns. The following **MATLAB one-based column indices** are used by the PI–porosity programs:

| Column | Variable | Use |
| --- | --- | --- |
| 1 | Depth | Well-log depth (m) |
| 9 | EI1 | Elastic impedance at 8°; network/inversion input |
| 10 | EI2 | Elastic impedance at 16°; network/inversion input |
| 12 | PI | Reference Poisson impedance; supervision/calibration and plotting |
| 13 | Porosity | Reference porosity (fraction); supervision/calibration and plotting |

The Python zero-based column indices are 0, 8, 9, 11, and 12, respectively. The supplied depth samples are in ascending order. The remaining MAT columns are not needed by the scripts in this module.

**Important:** The `Modeltest.mat` dataset is a separately supplied 19-column example matrix; the 15-column "Calculated_parameters.txt" generated in Module 01 is **not** automatically converted into this MAT file by the included scripts. The workflow therefore requires the supplied `Modeltest.mat`.


4. Software requirements
-----------
- Python 3 with numpy, scipy, pandas, matplotlib, scikit-learn, and a compatible version of tensorflow / Keras.
- MATLAB with the Statistics and Machine Learning Toolbox, required by fitgmdist and statset in the Bayesian baseline.
- A local copy of Modeltest.mat alongside the scripts, or an appropriate current working directory.

Example Python installation:

```bash
python -m pip install numpy scipy pandas matplotlib scikit-learn tensorflow
```

Exact dependency versions, a random seed for the GRU scripts, and platform-specific GPU requirements are **not pinned in the supplied implementation**. Consequently, GRU outputs may differ across runs or environments.

5. Execution procedure
-----------
Run the programs from the Module 02 directory so that their output CSV files can be located by well_compare_view.py.

Step 1 — Adapt and run the conventional Bayesian baseline

In MATLAB, execute:

matlab
PI_por_statistical_Bayesian_inversion


Expected CSV output:

PI_por_Bayesian_results.csv


The baseline calibrates a ridge-regularized statistical EI relationship and a three-component Gaussian-mixture prior; it generates 10,000 Monte Carlo samples and estimates marginal maximum-a-posteriori PI and porosity values. Its random seed is set to `2026`.

Step 2 — Run the data-driven GRU baseline (No RPC)

bash
python PI_Por_prediction_GRU_DD.py


Expected CSV output:

prediction_data_driven.csv


The network learns PI and porosity directly from the two EI inputs, with no reflectivity-based physical penalty.

Step 3 — Run the rock-physics-constrained GRU (With RPC)

bash
python PI_Por_prediction_GRU_RPC.py


Expected files:

prediction_physics_constrained.csv
best_GRU_PI_Por_v8_onestage.keras


The `.keras` file is saved whenever **training loss** improves. The CSV, however, is generated from the **in-memory weights at the end of training**, which are not explicitly reloaded from the saved checkpoint. These outputs may therefore correspond to different training epochs.

Step 4 — Plot and compare the three methods

After all three CSV files have been generated in the same directory, run:

bash
python well_compare_view.py


The comparison script creates:

PI_Por_a.tiff
PI_Por_b.tiff
PI_Error_Comparison.tiff
Porosity_Error_Comparison.tiff


It also prints error statistics, including MAE, MSE, RMSE, R², Pearson correlation, Spearman correlation, and mean error, together with method rankings. The script expects **identically ordered and aligned depth samples** in all three CSV files; it does not join rows by depth. 

6. GRU architecture and training configuration
-----------
Both supplied GRU scripts use the following basic architecture:

Input: EI1, EI2
  → Dense(128, ReLU, L2 kernel regularization)
  → GRU(128, dropout=0.2)
  → Dense(2, sigmoid)
Output: normalized PI and porosity

Inputs are reshaped as (number_of_samples, 2, 1); the two GRU sequence positions correspond to the **two EI angles**. Both scripts apply "MinMaxScaler" fitted to the complete example dataset.

| Configuration | No RPC GRU | With RPC GRU |
| --- | --- | --- |
| Optimizer | RMSprop | RMSprop |
| Maximum learning rate | 0.01 | 0.01 |
| Minimum learning rate | 1e-5 | 1e-5 |
| Warm-up steps | 100 | 100 |
| Cosine period parameter | 3000 optimizer steps | 3000 optimizer steps |
| Gradient clipping | clipnorm=1.0 | clipnorm=1.0 |
| Training limit | 1000 epochs | 1000 epochs |
| Training | Keras `model.fit`, default minibatching | Manual full-batch gradient updates |
| Model objective | MSE on normalized PI and porosity | Weighted data losses plus physical reflectivity loss |
| Data partition | Complete dataset | Complete dataset |



7. Interpretation and reproducibility notes
-----------
- The three methods are evaluated on the supplied example dataset. These are **controlled same-dataset comparisons**.
- The Bayesian baseline and the GRU approaches have different calibration/training procedures. The presence or absence of RPC can be evaluated more directly by comparing the two GRU implementations.
- Running the scripts does not itself establish the transferability of the model to other geological formations.

8. Place in the complete repository
-----------
This is **Module 02: Physics-Constrained PI–Porosity Prediction**.  In the synthetic module-level experiments, this task can be tested using reference EI inputs without propagating errors from other prediction models. 
