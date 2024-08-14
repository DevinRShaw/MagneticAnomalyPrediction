
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
    '3_gl_tot_sed_thick_',         # Total sediment thickness in the geological layer.
    '6_interpolated_bouguer_',     # Interpolated Bouguer gravity anomaly data.
    '5_gl_elevation_',             # Elevation of the ground level relative to sea level.
    '9__igrf_dec_',                # Declination (angle between magnetic north and true north) from the IGRF model.
    '4_cm_curie_point_depth_'      # Depth at which magnetic minerals lose their magnetism (Curie point).
]
```

## Clustering 
```python
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_features = ['6_interpolated_bouguer_','4_cm_curie_point_depth_']
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
exclude_columns = ['EMAG2v3','Latitude','Longitude','1_interpolated_mf7_','2_interpolated_emm_']
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
| Mean Squared Error (MSE)             | 808.6907547718187      |
| R<sup>2</sup> Score                  | 0.7604477603469891     |
| Root Mean Squared Error (RMSE)       | 28.437488545436263      |
| Mean Absolute Error (MAE)            | 19.54092916704921      |
| Coefficient of Variation of RMSE     | 0.489440741717535      |



![download](https://github.com/user-attachments/assets/b461ae0a-efdb-4c41-aedb-63adde66f07b)



### Test Areas Performance


| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 5642.523807946608      |
| R<sup>2</sup> Score                  | -1.9263026724438173      |
| Root Mean Squared Error (RMSE)       | 75.11673453995859    |
| Mean Absolute Error (MAE)            |  60.41685155436427      |
| Coefficient of Variation of RMSE     | 1.710643935026754      |

![download](https://github.com/user-attachments/assets/9155ac12-2ccd-4388-99c4-2489a46148bc)



| Predicted               | Actual                | EMM                  | MF7 | Cluster 
|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| ![image](https://github.com/user-attachments/assets/2e233122-cfdc-4f40-be87-5c16b928d36e)| ![download](https://github.com/user-attachments/assets/2e20eab1-d601-4bc3-99ee-111b14a858f8)| ![image](https://github.com/user-attachments/assets/e97cc88b-8e48-44a4-ba93-2c52d684b624) |![download](https://github.com/user-attachments/assets/7af24d27-a3ca-474f-ac4b-8ea8b4661708)|![download](https://github.com/user-attachments/assets/7b6af43a-2f9a-4a78-a4c0-661633d8228a) |




## Feature Importances 

### Random Forest Importances 
| Feature                    | Importance |
|------------------------------------|----------------|
| 5_gl_elevation_                    | 0.2614         |
| 9__igrf_dec_                       | 0.2454         |
| 3_gl_tot_sed_thick_                | 0.1654         |
| cluster                            | 0.1638         |
| 6_interpolated_bouguer_            | 0.1523         |
| 4_cm_curie_point_depth_            | 0.0116         |


![download](https://github.com/user-attachments/assets/8bc8eb80-3ad9-419d-9099-2e3906ee2248)


