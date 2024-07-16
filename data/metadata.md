# Metadata for Magnetic Anomaly Prediction Dataset

## Dataset Overview
Title: Magnetic Anomaly Prediction Dataset
Description: This dataset contains global geophysical variables and corresponding magnetic anomaly values used for predicting magnetic anomalies as part of the MagNav project. The dataset is divided into original and standardized versions, with each file containing one predictor variable.

## Data Sources
- National Oceanic and Atmospheric Administration (NOAA)


## Data Files

### Original Data
- Archive Name: `original_data.zip`
  Description: Contains raw magnetic anomaly data and individual geophysical predictor variables as collected from various sources.
  - `original_data/magnetic_anomalies.csv`
    - Description: Raw magnetic anomaly data.
    - Fields:
      - `Latitude` (float): Latitude of the data point.
      - `Longitude` (float): Longitude of the data point.
      - `Target` (float): Measured magnetic anomaly value at the data point.
  - `original_data/predictor_1.csv`
    - Description: Raw data for Predictor 1.
    - Fields:
      - `Latitude` (float): Latitude of the data point.
      - `Longitude` (float): Longitude of the data point.
      - `Predictor 1` (float): First geophysical predictor variable.
  - `original_data/predictor_2.csv`
    - Description: Raw data for Predictor 2.
    - Fields:
      - `Latitude` (float): Latitude of the data point.
      - `Longitude` (float): Longitude of the data point.
      - `Predictor 2` (float): Second geophysical predictor variable.
  - ...
  - `original_data/predictor_n.csv`
    - Description: Raw data for Predictor n.
    - Fields:
      - `Latitude` (float): Latitude of the data point.
      - `Longitude` (float): Longitude of the data point.
      - `Predictor n` (float): nth geophysical predictor variable.

### Standardized Data
- Archive Name: `standardized_data.zip`
  Description: File standardization process involved renaming all files to intuitive naming and formatted .nc files to have same xarray metadata and variable naming. This makes the data easier to use for future work.

## Usage Notes
- The original data should be used for reference and comparison purposes.
- The standardized data is recommended for use in machine learning tasks due to its consistent format and normalization.

