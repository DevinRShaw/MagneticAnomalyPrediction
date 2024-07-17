# Metadata for Magnetic Anomaly Prediction 
**Data can be downloaded as zip files in the [source data releases](https://github.com/DevinRShaw/MagneticAnomalyPrediction/releases)**
## Data Overview
Data contains global geophysical variables and corresponding magnetic anomaly values used for predicting magnetic anomalies as part of the MagNav project. The data is divided into original and standardized versions, with each .nc file in either zip containing one predictor variable.

Benchmark datasets are csv files created from the data in files for input to the model and are created with the `regression_evaluation/grid_selection.ipnb` notebook.

## Original Data Sources
- Archive Name: `original_data.zip`

### [Predicting Marine In Situ Heat Flow Using a Geospatial Machine Learning Conformal Prediction](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GC010913)
The initial approach of this project follows this paper and many predictors used in model are sourced from the paper's supplementary data.
* Multiple predictors all of which are .nc files of 100 km<sup>2</sup> equal area resolution (EASE grid)
  - [link to data](https://figshare.com/articles/dataset/Data_and_supplemental_material_for_Predicting_marine_in-situ_heat_flow_using_a_geospatial_machine_learning_conformal_prediction_/22104830)
### National Oceanic and Atmospheric Administration (NOAA)
* Earth Magnetic Anomaly Grid (EMAG2 v3, Meyer et al., 2017, 2 arc-minute equiangular grid)
  - [link to data](https://www.ncei.noaa.gov/products/earth-magnetic-model-anomaly-grid-2)
  - `EMAG2v3onPredictorMesh.csv` or `interpolated_emag_from_csv.nc` converted to csv then interpolated to new resolution CSV via Oasis Montaj
  
### International Gravimetric Bureau (BGI)
* Bouguer Gravity Anomaly
  - [link to data](https://bgi.obs-mip.fr/catalogue/?uuid=df2dab2d-a826-4776-b49f-61e8b284c409)
  - `interpolated_bouguer_from_csv.nc` converted to csv then interpolated to new resolution CSV via Oasis Montaj

 
### Magnetic Declination/Inclination, EMM, MF7 Sources
* Magnetic Declination/Inclination 
  - [link to data](https://bgi.obs-mip.fr/catalogue/?uuid=df2dab2d-a826-4776-b49f-61e8b284c409)
  - `igrf_dec.nc` and `igrf_inc.nc` converted to an .nc file via {insert method used here}

## Standardized Data
- Archive Name: `standardized_data.zip`
  
File standardization involves renaming all files to intuitive naming and formatted .nc files to have same xarray metadata and variable naming. This makes the data easier to use for future work.


## Usage Notes
- The original data should be used for reference and comparison purposes.
- The standardized data is recommended for use in machine learning tasks due to its consistent format and normalization.


