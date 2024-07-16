# Metadata for Magnetic Anomaly Prediction Data

## Dataset Overview
Title: Magnetic Anomaly Prediction Dataset
Description: Data contains global geophysical variables and corresponding magnetic anomaly values used for predicting magnetic anomalies as part of the MagNav project. The data is divided into original and standardized versions, with each file containing one predictor variable.

## Original Data Sources
- Archive Name: `original_data.zip`
### National Oceanic and Atmospheric Administration (NOAA)
* Earth Magnetic Anomaly Grid (EMAG2 v3, Meyer et al., 2017, 2 arc-minute equiangular grid) [link](https://www.ncei.noaa.gov/products/earth-magnetic-model-anomaly-grid-2)
** herhe
* 



## Standardized Data
- Archive Name: `standardized_data.zip`
File standardization involves renaming all files to intuitive naming and formatted .nc files to have same xarray metadata and variable naming. This makes the data easier to use for future work.

## Benchmark Data

## Usage Notes
- The original data should be used for reference and comparison purposes.
- The standardized data is recommended for use in machine learning tasks due to its consistent format and normalization.

