# Metadata for Magnetic Anomaly Prediction Dataset

## Dataset Overview
Title: Magnetic Anomaly Prediction Dataset
Description: This dataset contains global geophysical variables and corresponding magnetic anomaly values used for predicting magnetic anomalies as part of the MagNav project. The dataset is divided into original and standardized versions, with each file containing one predictor variable.

## Data Sources
- Source: National Oceanic and Atmospheric Administration (NOAA)
- Collection Period: January 2020 - December 2023
- Geographic Coverage: Global

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
  Description: Contains processed and standardized versions of the original data, suitable for machine learning tasks, with each file containing one predictor variable.
  - `standardized_data/magnetic_anomalies_standardized.csv`
    - Description: Standardized magnetic anomaly data.
    - Fields:
      - `Latitude` (float): Latitude of the data point.
      - `Longitude` (float): Longitude of the data point.
      - `Target` (float): Standardized magnetic anomaly value at the data point.
  - `standardized_data/predictor_1_standardized.csv`
    - Description: Standardized data for Predictor 1.
    - Fields:
      - `Latitude` (float): Latitude of the data point.
      - `Longitude` (float): Longitude of the data point.
      - `Predictor 1` (float): Standardized first geophysical predictor variable.
  - `standardized_data/predictor_2_standardized.csv`
    - Description: Standardized data for Predictor 2.
    - Fields:
      - `Latitude` (float): Latitude of the data point.
      - `Longitude` (float): Longitude of the data point.
      - `Predictor 2` (float): Standardized second geophysical predictor variable.
  - ...
  - `standardized_data/predictor_n_standardized.csv`
    - Description: Standardized data for Predictor n.
    - Fields:
      - `Latitude` (float): Latitude of the data point.
      - `Longitude` (float): Longitude of the data point.
      - `Predictor n` (float): Standardized nth geophysical predictor variable.


## Data Preprocessing
### Missing Values
- Missing values in the benchmark data were imputed using median imputation.

### Normalization
- Predictor variables in the standardized data were normalized using Min-Max scaling to ensure all features are within the same range.

### Standardization Process
- The standardization process involved transforming the raw data into a consistent format, normalizing predictor variables, and handling missing values to prepare the dataset for machine learning model training and evaluation.

## Usage Notes
- The original data should be used for reference and comparison purposes.
- The standardized data is recommended for use in machine learning tasks due to its consistent format and normalization.

