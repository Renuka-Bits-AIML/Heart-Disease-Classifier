# Dataset source

This project uses the processed Cleveland subset of the UCI Heart Disease dataset.
The official dataset page describes four source databases and identifies Cleveland
as the subset used in most published experiments. Its 303 rows have 13 predictors
and a diagnosis value (`num`) from 0 through 4. This project maps 0 to no detected
disease and values 1 through 4 to detected disease.

- UCI page: https://archive.ics.uci.edu/dataset/45/heart+disease
- Official download archive: https://archive.ics.uci.edu/static/public/45/heart+disease.zip
- DOI: https://doi.org/10.24432/C52P4X
- Citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
  Heart Disease [Dataset]. UCI Machine Learning Repository.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).

The training command downloads `processed.cleveland.data` from the official UCI
archive when the local raw file is absent. The source file uses `?` for missing
values; the data loader preserves these as missing values for in-pipeline imputation.