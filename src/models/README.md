# Model code

The AR, Ridge, and LightGBM leads create and maintain their own model files, covering model-specific preprocessing, parameter selection, training, prediction, and saving. Use the common data and time splits; output consistent prediction tables with log returns in y_true and y_pred. Fit learned preprocessing only on each training window. Agree filenames and interfaces within the team; no code is pre-created.
