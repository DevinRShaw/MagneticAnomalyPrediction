# Train/Test Grid Selection 
Grid selection notebook in ```prediction_evaluation/grid_selection.ipynb```. Due to the size of our files and large areas of missing data, it is faster to train/test on boundary boxes. Takes .nc files and creates a CSV file where each (lat,lon) combination is represented in (row = sample) and (column = feature) format.

## Benchmark CSV Format 
A row = sample, column = feature format allows for train/test data to be moved to the benchmark notebook, ```prediction_evaluation/random_forest_benchmark```, and simplifies input to the model via pandas/Numpy compatibility.

### Example of Format

| Latitude       | Longitude      | Target         | Predictor 1     | Predictor 2     | ...  | Predictor n     |
|----------------|----------------|----------------|-----------------|-----------------|------|-----------------|
| Sample 1, Latitude | Sample 1, Longitude | Sample 1, Target | Sample 1, Predictor 1 | Sample 1, Predictor 2 | ...  | Sample 1, Predictor n |
| Sample 2, Latitude | Sample 2, Longitude | Sample 2, Target | Sample 2, Predictor 1 | Sample 2, Predictor 2 | ...  | Sample 2, Predictor n |
| Sample 3, Latitude | Sample 3, Longitude | Sample 3, Target | Sample 3, Predictor 1 | Sample 3, Predictor 2 | ...  | Sample 3, Predictor n |
| ...            | ...            | ...            | ...             | ...             | ...  | ...             |
| Sample n, Latitude | Sample n, Longitude | Sample n, Target | Sample n, Predictor 1 | Sample n, Predictor 2 | ...  | Sample n, Predictor n |



# Region/Hole Benchmarks 
To see if the model can accurately predict within the geographic area it is trained on, we are training on 10 x 10 degree regions with 1 x 1 degree areas/holes missing, then comparing model predictions of areas with the actual values in the area. The idea is that if area is important, the model will model the areas missing from training data well. Feature importance was also investigated in these benchmarks to see if feature contributions to random forest change by area. 

Due to magnetic activity, an Atlantic regions, with 3 areas for testing was selected

## Atlantic Training Region Plot 
![download](https://github.com/user-attachments/assets/45d4d913-64b6-47f5-b344-f420da5b5719)

## Atlantic Region Boxes 
```python
region = [(38, -25, 48, -15)]

holes = [
    (39,-23,40,-22),
    (42,-20, 43,-19),
    (43,-17,44,-16)
]
```
## Feature Selection 
### Feature Rankings
![download](https://github.com/user-attachments/assets/81eb3a33-ef8f-4306-9ee1-61ced0643ccc)

### Predictors Trained On
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```.
```python
features = [
    'cluster',                  # Cluster identifier grouping data points with similar properties.
    'Longitude',                # Geographical coordinate specifying the east-west position.
    'Latitude',                 # Geographical coordinate specifying the north-south position.
    '3_gl_tot_sed_thick_',      # Total sediment thickness in the geological layer.
    '10_sc_crust_vs_',          # Shear wave velocity in the crustal layer.
    '4_cm_curie_point_depth_',  # Depth at which magnetic minerals lose their magnetism (Curie point).
    '1_interpolated_mf7_',      # Interpolated magnetic field data from the MF7 model.
    '2_interpolated_emm_',      # Interpolated magnetic field data from the EMM model.
    '6_interpolated_bouguer_',  # Interpolated Bouguer gravity anomaly data.
    '9__igrf_dec_',             # Declination (angle between magnetic north and true north) from the IGRF model.
    '5_gl_elevation_',          # Elevation of the ground level relative to sea level.
    '7_sc_crust_vp_',           # P-wave velocity in the crustal layer.
    '8_sc_crust_den_'           # Density of the crustal layer.
]
```



# Model Training 

A random forest regressor was trained on data with normalized features and median missing value imputation. The same training and hyperparameters were used for both benchmarks.
```python
kmeans = KMeans(n_clusters=10, random_state=42)
data['cluster'] = kmeans.fit_predict(data[['Latitude','Longitude']])

# Columns to exclude
exclude_columns = ['EMAG2v3']
include_columns = ['cluster','Longitude','Latitude']



for column in data.columns:
  if column in include_columns or column == 'EMAG2v3' or column in exclude_columns:
    continue
  try:
    if 0 < int(column[0:2]) and int(column[0:2]) <= 10: #select 5 highest ranked
      include_columns.append(column)
  except:
    if 0 < int(column[0]) and int(column[0]) <= 10: #select 5 highest ranked
      include_columns.append(column)
    continue


# Select columns not in exclude_columns using boolean indexing
X = data.loc[:, ~data.columns.isin(exclude_columns)]
X = X.loc[:, X.columns.isin(include_columns)]

X = X.fillna(X.median())


y = data['EMAG2v3']

y = y.fillna(y.median())
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=500, random_state=42,oob_score=True)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

```


# Evaluation 

## Metrics 

### Explanation of Metrics

- **Mean Squared Error (MSE)**: This metric measures the average squared difference between the predicted values and the actual values. It is useful for understanding the overall performance of the model. A lower MSE indicates better model performance.

- **R<sup>2</sup> Score**: Also known as the coefficient of determination, this metric indicates how well the model's predictions fit the actual data. It ranges from 0 to 1, with values closer to 1 indicating a better fit. An R<sup>2</sup> score of 0.8477694942863158 means that approximately 84.77% of the variance in the target variable is explained by the model.

- **Root Mean Squared Error (RMSE)**: This is the square root of the Mean Squared Error and provides a measure of the average magnitude of the prediction errors. RMSE is in the same units as the target variable, making it easier to interpret. A lower RMSE indicates better model performance.

- **Mean Absolute Error (MAE)**: This metric measures the average absolute difference between the predicted values and the actual values. Like RMSE, MAE provides a measure of the average magnitude of the errors, but it is less sensitive to large errors. A lower MAE indicates better model performance.

- **Coefficient of Variation of RMSE (CVRMSE)**: This metric is the RMSE divided by the mean of the observed data, expressed as a percentage. It provides a normalized measure of the prediction error, allowing for comparison across different datasets. A lower CVRMSE indicates better model performance.


### Region Training Metrics

| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 361.24153146259425      |
| R<sup>2</sup> Score                  | 0.8557001921109855     |
| Root Mean Squared Error (RMSE)       | 19.0063550283213      |
| Mean Absolute Error (MAE)            | 14.251835213255994      |
| Coefficient of Variation of RMSE     | 0.3798681453991826      |


![download](https://github.com/user-attachments/assets/97bf1b5c-2933-4c63-a7af-d9d2e8f06e65)


### Test Areas Performance


| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 869.7331925019588      |
| R<sup>2</sup> Score                  | 0.74524492413349      |
| Root Mean Squared Error (RMSE)       | 29.491239250020655      |
| Mean Absolute Error (MAE)            |  23.3826499829736      |
| Coefficient of Variation of RMSE     | 0.5047326776289702      |


![download](https://github.com/user-attachments/assets/9a92e420-dde7-4b13-a704-98f2600f7be1)

| Predicted               | Actual                |
|-----------------------|-----------------------|
| ![download](https://github.com/user-attachments/assets/8cd221ff-1e97-4ba2-96d0-87032c431b19) | ![download](https://github.com/user-attachments/assets/81b8cf71-e5e7-47db-b9b8-aa2d6a724de5) |
|  ![download](https://github.com/user-attachments/assets/25f64f8f-832c-48ef-8df3-d0e2e59f081d) |![download](https://github.com/user-attachments/assets/13c41aa0-01eb-4dc3-a4aa-eeb4023b0b3b) |
|![download](https://github.com/user-attachments/assets/b9bf3fc4-bcfc-47f0-ba20-a6eaf16a80b6) | ![download](https://github.com/user-attachments/assets/4518c349-eea4-4214-b253-fd709b6f6c35) |


## Feature Importances 

### Random Forest Importances 
| Feature                        | Importance |
|--------------------------------|------------|
| 1_interpolated_mf7_            | 0.3912     |
| 6_interpolated_bouguer_        | 0.1651     |
| 4_cm_curie_point_depth_        | 0.0728     |
| 10_sc_crust_vs_                | 0.0582     |
| Longitude                      | 0.0470     |
| Latitude                       | 0.0460     |
| cluster                        | 0.0415     |
| 3_gl_tot_sed_thick_            | 0.0395     |
| 9__igrf_dec_                   | 0.0375     |
| 2_interpolated_emm_            | 0.0347     |
| 7_sc_crust_vp_                 | 0.0290     |
| 5_gl_elevation_                | 0.0260     |
| 8_sc_crust_den_                | 0.0115     |

![download](https://github.com/user-attachments/assets/929911d8-9f25-4c59-b301-a38be414c4c2)



