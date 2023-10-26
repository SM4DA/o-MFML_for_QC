# Optimized Multi-Fidelity Machine Learning for Quantum Chemistry

This repository contains the scripts and data to reproduce the results of the work by Vinod _et. al._ titled "Optimized Multi-Fidelity Machine Learning for Quantum Chemistry. The raw data of molecules for the QM7b dataset can be downloaded from [https://achs-prod.acs.org/doi/10.1021/acs.jctc.8b00832#article_content-right]. The rawdata for the Excitation State Energies can be downloaded from [https://github.com/SM4DA/MultiFidelityMachineLearning-for-MolecularExcitationEnergies] with explanation present in Vinod _et. al._ (2023) available at [https://pubs.acs.org/doi/10.1021/acs.jctc.3c00882].

The scripts in this repository and the plots they reproduce are listed below:
* `QM7b/GenerateSLATM.py` generates the Global SLATM representation for the 7211 molecules of the QM7b data.
* `QM7b/LearningCurves_QM7b.py` generates data to reproduce Figure 3-5 of the main manuscript and Figure 1 of the supplementary text.
* `QM7b/pople_MFML_outs.py` generates the single fidelity learning curve from these figures.
* `QM7b/Coeff_analysis_removed_fidelity.py` compares the full o-MFML model and reduced o-MFML model as per the analysis of hte coefficients.
* `ExcitedState/LearningCurves_ExcitedState.py`  generates data for Figure 6,7 of the main text, and Figure 2,3 of the Supplementary text.
* `ExcitedState/CompareMFMLtypes.py` generates data for Table 1 in the supplementary text.

All the plotting routines for the QM7b segment are found in `QM7b/QM7bPlots.ipynb` and those for the Excitation state can be found in `ExcitedState/ExcitedStatePlots.ipynb`.
