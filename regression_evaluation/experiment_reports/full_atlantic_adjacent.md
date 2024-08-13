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



# Region/Adjacent Benchmarks 
To see if the model can accurately predict around the geographic area it is trained on, we are training on a 10 x 20 degree region, then testing model accuracy on an adjacent region to see if larger scale patterns are captured. Feature importance was also investigated in these benchmarks to see if feature contributions to random forest change by area. 

Due to magnetic activity, an Atlantic region was selected.

## Atlantic Training Region Plot 
![download](https://github.com/user-attachments/assets/ef4459c0-5b45-4bec-b3f6-2b804ca00fb0)

## Atlantic Testing Region Plots
![download](https://github.com/user-attachments/assets/37f29c6b-b586-4215-9dd9-c77594389e90)

## Atlantic Region Boxes 
```python
region = [(28, -25, 48, -15)] # region to be train model 
adjacent_region = [(38, -14, 48, -4)] # region to test model 
```
## Feature Selection 
### Feature Rankings
![download](https://github.com/user-attachments/assets/81eb3a33-ef8f-4306-9ee1-61ced0643ccc)

### Predictors Trained On
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. Features relating to 'crust' were dropped for this experiment.

```python
features = [
    'cluster',                     # Cluster identifier grouping data points with similar properties. (e.g., curie and bouguer)
    '1_interpolated_mf7_',         # Interpolated magnetic field data from the MF7 model.
    '3_gl_tot_sed_thick_',         # Total sediment thickness in the geological layer.
    '2_interpolated_emm_',         # Interpolated magnetic field data from the EMM model.
    '6_interpolated_bouguer_',     # Interpolated Bouguer gravity anomaly data.
    '5_gl_elevation_',             # Elevation of the ground level relative to sea level.
    '9__igrf_dec_',                # Declination (angle between magnetic north and true north) from the IGRF model.
    '4_cm_curie_point_depth_'      # Depth at which magnetic minerals lose their magnetism (Curie point).
]
```

## Clustering 
```python
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_features = []
cluster_features.append('6_interpolated_bouguer_')
cluster_features.append('4_cm_curie_point_depth_')
data['cluster'] = kmeans.fit_predict(data[cluster_features])
```

# Model Training 

A random forest regressor was trained on data with normalized features and median missing value imputation. The same training and hyperparameters were used for both benchmarks.
```python
kmeans = KMeans(n_clusters=10, random_state=42)

cluster_features = []
#cluster_features = [column for column in data.columns if "crust" in column or 'thick' in column]
cluster_features.append('6_interpolated_bouguer_')
cluster_features.append('4_cm_curie_point_depth_')
data['cluster'] = kmeans.fit_predict(data[cluster_features])

# Columns to exclude
exclude_columns = ['EMAG2v3','Latitude','Longitude']
include_columns = ['cluster']


#watch out for crustal feature removal
for column in data.columns:
  if column in include_columns or column == 'EMAG2v3' or column in exclude_columns or 'crust' in column:
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

'''
# Normalize the features
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
'''

y = data['EMAG2v3']

y = y.fillna(y.median())
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
| Mean Squared Error (MSE)             | 494.58969802452384      |
| R<sup>2</sup> Score                  | 0.8534914994737246     |
| Root Mean Squared Error (RMSE)       | 22.239372698539047      |
| Mean Absolute Error (MAE)            | 15.83382912088798      |
| Coefficient of Variation of RMSE     | 0.38276428846781846      |


![download](https://github.com/user-attachments/assets/3f750b6c-7d48-4b73-8fd4-c6fe691e29e7)



### Test Areas Performance


| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 1576.0091947742508      |
| R<sup>2</sup> Score                  | 0.18265654245555685      |
| Root Mean Squared Error (RMSE)       | 39.69898228889817     |
| Mean Absolute Error (MAE)            |  29.518793449805052      |
| Coefficient of Variation of RMSE     | 0.9040704936809094      |

![download](https://github.com/user-attachments/assets/9155ac12-2ccd-4388-99c4-2489a46148bc)



| Predicted               | Actual                | EMM
|-----------------------|-----------------------|-----------------------|
| ![download](https://github.com/user-attachments/assets/4df2f625-46a1-4a29-a2c3-855265064242) | ![download](https://github.com/user-attachments/assets/2e20eab1-d601-4bc3-99ee-111b14a858f8)| ![image](https://github.com/user-attachments/assets/e97cc88b-8e48-44a4-ba93-2c52d684b624) |



## Feature Importances 

### Random Forest Importances 
| Feature                        | Importance |
|--------------------------------|------------|
| cluster                        | 0.3066     |
| 3_gl_tot_sed_thick_            | 0.2630     |
| 5_gl_elevation_                | 0.1142     |
| 9__igrf_dec_                   | 0.1005     |
| 1_interpolated_mf7_            | 0.0850     |
| 2_interpolated_emm_            | 0.0648     |
| 6_interpolated_bouguer_        | 0.0570     |
| 4_cm_curie_point_depth_        | 0.0088     |


![download](https://github.com/user-attachments/assets/bab6a42a-8cd3-43c2-9666-efc04548a17f)

## Cluster Analysis 
Scatter plots and distributions of features used for clustering and target, with data labeled by kmeans cluster.

![download](https://github.com/user-attachments/assets/224d4b30-d8f6-444d-8b38-a64e8ec74344)




