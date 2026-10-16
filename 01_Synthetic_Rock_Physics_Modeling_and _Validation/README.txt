MODULE 01: SYNTHETIC ROCK-PHYSICS MODELING AND VALIDATION
========================================================

1. OVERVIEW
-----------
This module provides a reproducible workflow for rock-physics forward
modeling, elastic-attribute calculation, lithology-specific velocity
comparisons, and well-log visualization. It supports the synthetic-model
study in the manuscript "Physics-Constrained Intelligent Inversion and
Marine Caprock Identification Driven by Rock Physics".

The main forward model calculates P-wave and S-wave velocities from mineral
fractions, porosity, pore parameters, effective pressure, fluid saturation,
and density. The MATLAB analysis compares these velocities with reference
velocity logs. Derived elastic impedance (EI) and Poisson impedance (PI)
attributes, lithology-specific crossplots, and individual log plots are
produced for inspection and use in later modules.

This module is a forward-modeling example. 
It is NOT an independent train/test experiment and does not train
any neural networks.

2. FILES
--------
Model_data_used_to_Rock_physics_forward.txt
    Input dataset containing 16 columns of depth-indexed petrophysical and
    elastic properties.

Rock_physics_forward.m
    MATLAB function implementing the rock-physics forward calculation.
    IMPORTANT: Its function is named rock_physics_forward, so rename or copy
    this file to rock_physics_forward.m before running the main MATLAB script.

Calculate_Elastic_parameter_main.m
    Main MATLAB script. Calls the forward function; computes modeled P- and
    S-wave velocities, velocity errors, EI at 8, 16, and 24 degrees, and PI;
    exports Calculated_parameters.txt and figures.

Lithology_specific_validation.m
    MATLAB script that reads Calculated_parameters.txt and plots observed
    versus modeled P-wave and S-wave velocity crossplots, grouped by
    lithology.

Well_curves_plot.py
    Python plotting script that reads Calculated_parameters.txt and saves
    12 separate, depth-aligned log/attribute figures in TIFF format.
    This is a visualization script; it does not calculate the forward model.

3. SOFTWARE REQUIREMENTS
------------------------
MATLAB:
    - MATLAB (the forward function indicates MATLAB R2019b compatibility).
    - Statistics and Machine Learning Toolbox for histfit used in the main
      script.

Python:
    - Python 3 with NumPy, pandas, Matplotlib, and Pillow (used for TIFF
      image handling/compression through Matplotlib).

Install Python dependencies, for example:

    python -m pip install numpy pandas matplotlib pillow

The exact environment versions have not been verified through a complete
runtime test; record actual versions when publishing reproducible results.

4. INPUT DATA FORMAT
--------------------
Input: Model_data_used_to_Rock_physics_forward.txt

Expected column order:

 1  depth       Depth (m)
 2  Pe          Effective pressure (MPa)
 3  vqua        Quartz volume fraction
 4  clay        Clay volume fraction
 5  vclc        Calcite volume fraction
 6  Feldspar    Feldspar volume fraction
 7  Dolomite    Dolomite volume fraction
 8  Kerogen     Kerogen fraction (not used in the present forward function)
 9  por         Porosity (fraction)
10  sw          Water saturation (fraction)
11  den         Bulk density (kg/m^3)
12  vp          Reference P-wave velocity (m/s)
13  VS          Reference S-wave velocity (m/s)
14  fsoft       Fraction of relatively compliant pore space
15  alpha       Pore aspect ratio
16  Lith        Lithology code

The example uses lithology codes 1, 2, and 3.
These are described as sandstone, mudstone, and calcareous sandstone, respectively. 

5. RUNNING THE WORKFLOW
-----------------------
Place all five source/input files in this module directory.


Step 1. Open MATLAB, set the current directory to this module folder, and run:

    run('Calculate_Elastic_parameter_main.m')

This produces:
    - Calculated_parameters.txt
    - figure/ directory containing velocity comparisons/error histograms
    - Velocity error statistics printed in the MATLAB Command Window

Step 2. Run the lithology-specific crossplot script in MATLAB:

    run('Lithology_specific_validation.m')

This reads Calculated_parameters.txt and produces:
    - Figure/Vp_prediction_crossplot.tif
    - Figure/Vs_prediction_crossplot.tif

Step 3. Run the Python well-curve visualization script:

    python Well_curves_plot.py

The script searches for Calculated_parameters.txt in its own directory or
in the current working directory and creates Figure/ automatically.

The steps must be run in this order because Steps 2 and 3 depend on
Calculated_parameters.txt generated in Step 1. Steps 2 and 3 are independent
of each other once that file exists.

6. INTERMEDIATE DATA FILE
-------------------------
Calculated_parameters.txt is produced by the MATLAB main script. It has
15 columns in this order:

Depth Vp_obs Vp_cal Vs_obs Vs_cal EI1 EI2 EI3 VQUA Vclay VCLC PI Por Lith RHOB

- Vp_obs, Vs_obs: reference velocities
- Vp_cal, Vs_cal: velocities computed by the rock-physics forward model
- EI1, EI2, EI3: elastic impedance attributes at 8, 16, and 24 degrees
- PI: Poisson impedance derived from reference velocities and density
- VQUA, Vclay, VCLC: mineral volume fractions
- Por: porosity fraction
- Lith: lithology category
- RHOB: bulk density (kg/m^3)

IMPORTANT: In the current MATLAB main script, EI and PI are derived from
the REFERENCE velocities and density, not from Vp_cal or Vs_cal.

7. OUTPUT FIGURES
-----------------
MATLAB main script (folder: figure/, lowercase):
    - Vp_obs_prediction.tif
    - Vs_obs_prediction.tif
    - Vp_obs_error_distribution_histogram_forward.tif
    - Vs_obs_error_distribution_histogram_forward.tif

MATLAB lithology validation (folder: Figure/, uppercase F):
    - Vp_prediction_crossplot.tif
    - Vs_prediction_crossplot.tif

Python Well_curves_plot.py (folder: Figure/, uppercase F):
    - PI.tiff           Poisson impedance
    - por.tiff          Porosity (%)
    - Vquar.tiff        Quartz fraction (%)
    - Vclc.tiff         Calcite fraction (%)
    - Vclay.tiff        Clay fraction (%)
    - Lith.tiff         Reference lithology color strip
    - Vp.tiff           Reference and modeled P-wave velocity
    - Vs.tiff           Reference and modeled S-wave velocity
    - Density.tiff      Bulk density
    - EI1.tiff          EI at 8 degrees
    - EI2.tiff          EI at 16 degrees
    - EI3.tiff          EI at 24 degrees

The Python script saves individual 600-dpi TIFF images, with LZW
compression requested through Matplotlib. Its horizontal axis limits are
configured near the beginning of Well_curves_plot.py.

NOTE: The MATLAB scripts use both 'figure/' and 'Figure/'. These may be
separate directories on case-sensitive operating systems. The Python
script's lithology-plot x-label currently says 'True lithology (Train)',
but this module has no training/validation split; edit this label to
'Reference lithology' for a public-facing figure if appropriate.

8. PHYSICAL MODEL AND IMPLEMENTATION DETAILS
--------------------------------------------
The forward model includes Voigt-Reuss-Hill mineral modulus averaging,
stiff/soft pore partitioning, aspect-ratio-dependent pore factors,
pressure-dependent pore stiffness, fluid-modulus handling, and saturated
elastic-velocity calculation.


9. VALIDATION SCOPE
-----------------------------------------
Observed-versus-modeled velocity plots quantify fit on the provided depth
interval. They do not independently establish generalization to unseen
wells or the absence of elastic anisotropy.


10. CONNECTION TO THE OTHER MODULES
-----------------------------------
Module 01 generates rock-physics reference quantities and elastic
attributes that support the later experiments:

02_Physics_Constrained_PI_Porosity_Prediction
03_Physics_Constrained_Mineral_Composition_Prediction
04_Rock_Physics_Guided_Lithology_Prediction
