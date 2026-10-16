Module 04 - Rock-Physics-Guided Lithology Prediction
===================================================

1. Overview
-----------
This module reproduces the synthetic-interval lithology classification experiment using two 1D-ResNet programs with different input attributes:

Conventional model: observed/reference P-wave velocity (Vp), S-wave velocity (Vs), and density (RHOB).
Rock-physics-guided model: reference Poisson impedance (PI), porosity, and quartz, calcite, and clay volume fractions.

Both models classify sandstone (1), mudstone (2), and calcareous sandstone (3). They use the same three-residual-block architecture, SGD optimizer, PS-CALR parameters, training limit, and depth-ordered 80%/20% partition. The shallowest 80% of samples are used for training, and the deepest continuous 20% are reserved for within-well validation.

This is Module 04, the lithology-classification stage of the staged marine caprock prediction workflow. In this example, the rock-physics-guided features are reference attributes, not predictions automatically transferred from Modules 02 and 03. There is no explicit differentiable rock-physics loss in either 1D-ResNet; the geological guidance is supplied through the input features.

2. Files
--------
Lithology_prediction_1DResNet_conventional.py
  Conventional 1D-ResNet classification program using Vp_obs, Vs_obs, and RHOB.

Conventional_data.csv
  Five-column input dataset for the conventional program.

Lithology_prediction_1DResNet_Rock_physics_guided.py
  Rock-physics-guided 1D-ResNet classification program using reference PI, porosity, and three mineral fractions.

Caprock sensitive parameters and mineral contents.csv
  Seven-column input dataset for the rock-physics-guided program.

README.md / README.txt
  Module instructions, configurations, output definitions, and interpretation notes.


3. Input data and depth partition
---------------------------------

Conventional_data.csv: 163 rows x 5 columns
Depth     Depth in meters
Vp_obs    P-wave velocity
Vs_obs    S-wave velocity
RHOB      Bulk density
Lith      Reference lithology class (1, 2, or 3)

Caprock sensitive parameters and mineral contents.csv: 163 rows x 7 columns
Depth        Depth in meters
True PI      Reference Poisson impedance
True Por     Reference porosity (fraction)
True Vquar   Reference quartz volume fraction
True Vclc    Reference calcite volume fraction
True Vclay   Reference clay volume fraction
Lith         Reference lithology class (1, 2, or 3)

Input feature vectors:
Conventional: [Vp_obs, Vs_obs, RHOB]
Rock-physics-guided: [True PI, True Por, True Vquar, True Vclc, True Vclay]

Depth is used for ordering and well-section plots, not as a predictor. Lith is the target variable, not a predictor. The lithology codes 1/2/3 are converted internally to the zero-based classes 0/1/2 for Keras and converted back when saving results. Porosity and mineral fractions are fractions rather than percentages.

In the latest supplied CSVs, the class counts are 134 sandstone, 9 mudstone, and 20 calcareous sandstone samples. With TEST_SIZE = 0.20, the depth-ordered holdout contains:

Training:   first 130 rows, 3726.1800 to 3745.8396 m
Validation: last 33 rows,  3745.9920 to 3750.8690 m

4. Software requirements
------------------------
Python 3 with compatible versions of TensorFlow/Keras, NumPy, pandas, Matplotlib, and scikit-learn.

Example installation:

    python -m pip install tensorflow numpy pandas matplotlib scikit-learn

Run each program with its matching CSV in the same directory. The source first searches beside the script and then in the current working directory. These scripts do not require MATLAB or saved weights from the preceding modules. Library versions are not pinned in the supplied files.

5. Execution procedure and outputs
----------------------------------
From the Module 04 directory, run the two programs independently.

Step 1 - Conventional-input 1D-ResNet

    python Lithology_prediction_1DResNet_conventional.py

Input: Conventional_data.csv
Output directory: Lithology_1DResNet_Conventional_PS_CALR_results/
Main prediction CSV: Lithology_prediction_1DResNet_Conventional_PS_CALR.csv
Additional TXT saved in the current working directory: Lithology_prediction_results_conventional.txt

Step 2 - Rock-physics-guided 1D-ResNet

    python Lithology_prediction_1DResNet_Rock_physics_guided.py

Input: Caprock sensitive parameters and mineral contents.csv
Output directory: Lithology_1DResNet_PS_CALR_Rock_physics_guided_results/
Main prediction CSV: Lithology_prediction_1DResNet_PS_CALR.csv
Additional TXT saved in the current working directory: Lithology_prediction_results_Rock_physics_guided.txt

Each program creates the following files inside its own output directory, in addition to its prediction CSV:

Classification_evaluation.txt
  Within-well validation accuracy, validation loss, confusion matrix, and precision/recall/F1 by class; training and validation depth ranges.
training_metrics_PS_CALR.csv
  Epoch-wise training and validation loss and accuracy.
best_1DResNet_PS_CALR.weights.h5
  Best weights selected by minimum validation loss.
Learning_Rate_PS_CALR.tiff
  Learning rate versus optimizer updates.
Training_Validation_Accuracy.tiff
  Accuracy curves.
Training_Validation_Loss.tiff
  Loss curves.
Lithology_True_vs_Predicted_1DResNet.tiff
  Full-depth reference/predicted lithology comparison.
True_Lithology_1D.tiff
  Full-depth reference lithology strip.
Predicted_Lithology_1D.tiff
  Full-depth predicted lithology strip.

Each prediction CSV includes Depth, True Lith, Pred Lith, Data Split, and three class probabilities. Data Split marks rows as Train or Validation. The additional three-column TXT file contains Depth, True_Lithology, and Predicted_Lithology. Plots are saved at 600 dpi, with LZW TIFF compression requested by Matplotlib.

The models run independently and write to different output folders, preventing checkpoint and figure name collisions. No fixed accuracy is guaranteed by these scripts.

6. Network and training configuration
-------------------------------------
Shared architecture:
  Input: (number_of_samples, number_of_features, 1)
  Conv1D: 64 filters, kernel size 3; batch normalization and ReLU
  Dropout: 0.5 after initial activation and between residual blocks
  Residual blocks: three blocks with 64, 128, and 128 channels
  Residual-branch convolutions: Conv1D with L2 kernel regularization
  GlobalAveragePooling1D followed by Dense(3) logits

Conventional input shape: (number_of_samples, 3, 1)
Rock-physics-guided input shape: (number_of_samples, 5, 1)

The Conv1D filters operate across the FEATURE sequence for each individual depth sample. The model does not use a window of adjacent depths as its input. The supervised objective is sparse categorical cross-entropy from logits, with regularization terms from the L2-constrained layers. There is no explicit physical consistency term in the classifier's loss.

Shared training configuration:
  Optimizer: SGD, no explicit clipnorm
  Maximum epochs: 600
  Batch size: 128
  Continuous-depth partition: 80% training / 20% validation
  StandardScaler: fitted to training rows only
  Maximum learning rate: 0.1
  Minimum learning rate: 0.01 (also the step-zero initial rate)
  Warm-up: 10 optimizer updates
  Cosine period parameter: 1200 optimizer updates
  EarlyStopping: validation loss, patience = 200 epochs
  Checkpoint: lowest validation loss; reloaded before prediction
  Seed value in both programs: 2026

The PS_CALR class implements optimizer-step-based linear warm-up followed by cosine annealing. 
The phase shift is implicitly set to zero in this implementation; consequently, no separate phase-shift parameter is required. 
The learning rate starts at 0.01, increases to a maximum of 0.1 during warm-up, and then follows the cosine schedule. 
These values refer to the initial/minimum and maximum learning rates, respectively.


7. Evaluation and reproducibility notes
---------------------------------------
The deepest 20% of samples form a contiguous, within-well validation segment. Full-depth outputs combine predictions on training and validation rows.

The two supplied programs have aligned inputs and the same main architecture, optimizer, PS-CALR settings, and validation split, but the random-control implementation differs:

Conventional script: tf.keras.utils.set_random_seed(2026) and tf.config.experimental.enable_op_determinism().
Rock-physics-guided script: tf.keras.utils.set_random_seed(2026) and tf.config.experimental.enable_op_determinism().

Consequently, the supplied programs are not identical in random-number and determinism configuration. Exact repeatability and a strictly matched feature-only comparison require harmonizing this setup and recording the Python/TensorFlow/hardware environment. No such harmonization is applied to the source files by this README.


8. Place in the complete repository
-----------------------------------
Module 01 provides rock-physics forward modeling and elastic-attribute validation. Module 02 estimates PI and porosity. Module 03 estimates mineral volume fractions. Module 04 compares conventional and rock-physics-guided lithology classification using aligned reference features in the synthetic example.

For a complete staged field application, the predicted PI, porosity, and mineral attributes from upstream modules would replace reference rock-physics-guided inputs; no automatic cross-module data transfer is implemented in this synthetic Module 04. 
