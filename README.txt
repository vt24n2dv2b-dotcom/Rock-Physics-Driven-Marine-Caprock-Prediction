Rock-Physics-Constrained Marine Caprock Prediction
==================================================

1. Overview
-----------
This repository contains MATLAB and Python research code and example synthetic-model data associated with the manuscript:

"Physics-Constrained Intelligent Inversion and Marine Caprock Identification Driven by Rock Physics"

The framework combines rock-physics forward modeling, physics-constrained prediction of Poisson impedance (PI) and porosity, prediction of mineral composition, and rock-physics-guided lithology identification. It is organized as four experimental modules. The code supplied here supports the synthetic-model experiments and method comparisons.

The module READMEs provide detailed instructions, exact input columns, scripts, training configurations, and interpretation limits. Please read the relevant module README before executing a workflow.

2. Repository structure
-----------------------
Rock_Physics_Constrained_Marine_Caprock_Prediction/
  README.txt

  01_Synthetic_Rock_Physics_Modeling_and_Validation/
    README.txt
    Rock_physics_forward.m
    Calculate_Elastic_parameter_main.m
    Lithology_specific_validation.m
    Well_curves_plot.py
    Model_data_used_to_Rock_physics_forward.txt

  02_Physics_Constrained_PI_Porosity_Prediction/
    README.txt
    Modeltest.mat
    PI_por_statistical_Bayesian_inversion.m
    PI_Por_prediction_GRU_DD.py.py
    PI_Por_prediction_GRU_RPC.py
    well_compare_view.py

  03_Physics_Constrained_Mineral_Composition_Prediction/
    README.txt
    Modeltest.mat
    Minerals_statistical_Bayesian_inversion.m
    Minerals_prediction_GRU_DD.py
    Minerals_prediction_GRU_RPC.py
    well_compare_view.py

  04_Rock_Physics_Guided_Lithology_Prediction/
    README.txt
    Lithology_prediction_1DResNet_conventional.py
    Conventional_data.csv
    Lithology_prediction_1DResNet_Rock_physics_guided.py
    Caprock sensitive parameters and mineral contents.csv

This is the INTENDED repository layout: copy the applicable code/data files into their corresponding folders before running the steps below. Modeltest.mat is used separately by Modules 02 and 03; it may be copied into both directories. Output directories and prediction files are generated at runtime and are not listed above.


3. Workflow and module roles
----------------------------
Module 01 - Synthetic rock-physics modeling and validation
  MATLAB rock-physics forward calculation of modeled Vp and Vs; comparison with reference logs; elastic impedance at 8, 16, and 24 degrees; PI; lithology crossplots and well-curve plots.

Module 02 - Physics-constrained PI and porosity prediction
  Comparison of statistical Bayesian inversion, data-driven GRU (No RPC), and rock-physics-constrained GRU (With RPC) using EI at 8 and 16 degrees.

Module 03 - Physics-constrained mineral composition prediction
  Comparison of statistical Bayesian inversion, data-driven GRU (No RPC), and rock-physics-constrained GRU (With RPC). Inputs include EI at 8, 16, and 24 degrees and reference PI/porosity where required by the GRU scripts; targets are quartz, calcite, and clay fractions.

Module 04 - Rock-physics-guided lithology prediction
  Comparison of two 1D-ResNet classifiers: conventional input [Vp, Vs, RHOB] versus rock-physics-guided input [PI, porosity, quartz, calcite, clay]. Both use the shallowest 80% for training and deepest continuous 20% for within-well validation.

The conceptual staged application is: elastic attributes -> PI and porosity -> mineral fractions -> lithology. For the supplied synthetic experiments, Modules 03 and 04 use available REFERENCE features to examine individual stages without propagated prediction errors. They are not connected into a single automated end-to-end run. The 15-column output from Module 01 is not automatically converted into the distinct 19-column Modeltest.mat used in Modules 02 and 03.

4. Software requirements
------------------------
MATLAB for Module 01 forward modeling and the Bayesian baselines in Modules 02 and 03. The Statistics and Machine Learning Toolbox is required by routines including histfit, fitgmdist, and statset.

Python 3 with NumPy, SciPy, pandas, Matplotlib, scikit-learn, and a compatible TensorFlow/Keras version. Pillow is useful for TIFF output in Module 01.

Example Python installation:

    python -m pip install numpy scipy pandas matplotlib scikit-learn tensorflow pillow

No tested environment lockfile or fixed dependency-version matrix is provided by the supplied material. GPU/CUDA/cuDNN details and TensorFlow deterministic-operation support may affect numerical reproducibility. 

5. Quick-start instructions
---------------------------
Run each module from its own folder after placing the named source/data files there.

Module 01:
  a. Rename 00_rock_physics_forward.m to rock_physics_forward.m so its MATLAB filename matches its function name.
  b. In MATLAB: run('01_Calculate_Elastic_parameter_main.m')
  c. In MATLAB: run('02_Lithology_specific_validation.m')
  d. In a terminal: python Well_curves_plot.py

Module 02:
  a. Confirm that the Bayesian MATLAB script loads Modeltest.mat and the variable Modeltest. The supplied script instead references PY1064_modeltest.mat / PY1064_modeltest, so its loading lines require adjustment before execution.
  b. In MATLAB: PI_por_statistical_Bayesian_inversion
  c. In a terminal: python PI_Por_prediction_GRU_DD.py.py
  d. In a terminal: python PI_Por_prediction_GRU_RPC.py
  e. In a terminal: python well_compare_view.py

Module 03:
  a. In MATLAB: Minerals_statistical_Bayesian_inversion
  b. In a terminal: python Minerals_prediction_GRU_DD.py
  c. In a terminal: python Minerals_prediction_GRU_RPC.py
  d. In a terminal: python "well_compare_view.py"

Module 04:
  a. In a terminal: python Lithology_prediction_1DResNet_conventional.py
  b. In a terminal: python Lithology_prediction_1DResNet_Rock_physics_guided.py

Modules 02 and 03 require their Bayesian and GRU result CSVs to be generated before running their respective plotting/comparison scripts. Module 04 runs its two classifiers independently and writes to separate result directories. Detailed output names and prerequisites are documented within each module README.
The repository documents numerical configurations as implemented by the supplied scripts, rather than assuming they exactly match every prose description or formula in the manuscript. The PS_CALR implementations use optimizer-step-based linear warm-up and cosine scheduling; where no phase-shift term is specified, the implemented phase shift is implicitly zero. A nonzero phase shift is not used in these example scripts.

7. Citation and contact
-----------------------
If using this code in research, cite the associated manuscript after bibliographic details are finalized:

"Physics-Constrained Intelligent Inversion and Marine Caprock Identification Driven by Rock Physics"

Please consult the module-specific source code and README for the corresponding method, inputs, and outputs. 
