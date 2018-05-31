### Gaussian process-based disease progression modeling and time-shift estimation.
#
#  - The software iteratively estimates monotonic progressions for each biomarker and realigns the individual observations in time
#   Basic usage:
#
#       model = GP_progression_model.GP_progression_model(input_X,input_N,N_random_features)
#
#   X and Y should be A list of biomarkers arrays. Each entry "i" of the list is a list of individuals' observations for the biomarker i
#   The monotonicity is enforced by the parameter self.penalty (higher -> more monotonic)
#
# - The class comes with an external method for transforming a given .csv file in the required input X and Y:
#
#       X,Y,list_biomarker = GP_progression_model.convert_csv(file_path)
#
# - The method Save(folder_path) saves the model parameters to an external folder, that can be subsequently read with the
# method Load(folder_path)
#
# - Optimization can be done with the method Optimize:
#
#       model.Optimize()
#
# - Examples on synthetic and on ADNI data are provided in the 'examples' folder  
#
### This software is based on the publication:
#
# Probabilistic disease progression modeling to characterize diagnostic uncertainty: Application to staging and prediction in Alzheimer's disease
# Marco Lorenzi, Maurizio Filippone, Giovanni B. Frisoni, Daniel C. Alexander, Sebastien Ourselin
# NeuroImage, 2018
# https://doi.org/10.1016/j.neuroimage.2017.08.059
#
# Gaussian process regression based on random features approximations is based on the paper:
#
# Random Feature Expansions for Deep Gaussian Processes (ICML 2017, Sydney)
# K. Cutajar, E. V. Bonilla, P. Michiardi, M. Filippone
# arXiv:1610.04386
#
#
### Authors
# Marco Lorenzi and Maurizio Filippone
