
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



# Small/Large Benchmarks 
Selecting for train/test data for chosen features and boundary boxes. A small dataset of size (36297 samples, 1 box) and a large dataset of size (356911 samples, 10 boxes) are created for input to the model. The smaller dataset is used to measure how much sample size improves model performance. The model was not trained on the entire globe mainly due to time to train taking too long for project time constraint.


## Feature Selection 

### Predictors Trained On
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. The goal of selecting features is to improve model performance without model overfitting. Simply choosing high ranked factors will not yield an optimal result due to learning training data too well to generalize. EMM and MF7 {insert scientific names here} were dropped due to having a much larger prediction power than other predictors, and their close domain relation to the target variable, both of which can cause overfitting. The next {insert the accurate k of kth best predictors that yielded best test}

| Predictors                  | Description                                                                            |
|-----------------------------|----------------------------------------------------------------------------------------|
| cm_curie_point_depth        | Depth to the Curie point, the temperature at which certain materials lose their permanent magnetic properties. |
| wgm2012_freeair_ponc        | Free-air gravity anomalies from the WGM2012 gravity model, which measure variations in Earth's gravity field.   |
| gl_elevation                | Elevation above sea level derived from global elevation datasets.                                            |
| rayleigh_group              | Group velocity of Rayleigh waves, which are surface seismic waves that cause the ground to shake in an elliptical motion. |
| sc_crust_den                | Density of the Earth's crust, typically measured in grams per cubic centimeter.                               |
| interpolated_bouguer        | Bouguer gravity anomalies corrected for the gravitational effect of topography.                                |
| igrf_dec                    | Magnetic declination from the International Geomagnetic Reference Field, indicating the angle between magnetic north and true north. |
| love_phase                  | Phase velocity of Love waves, which are surface seismic waves that cause horizontal shifting of the Earth during an earthquake. |
| gl_tot_sed_thick            | Total sediment thickness, which indicates the depth of sediment layers on the Earth's surface.                 |



## Small Benchmark Boundary Box
Both boundary box sets are defined as list of tuples becuase```filter_by_boundary_boxes(df, boundary_boxes)``` using a list of tuples for boundary boxes.
```python
boundary_box = [(-115,33,-83,43)]
```

## Large Benchmark Boundary Boxes 

```python
# Define the boundary boxes for selection
boundary_boxes = [
    # USA: (latitude range: 24N to 49N, longitude range: 125W to 66W)
    (24, -125, 49, -66),    # Box 1: from (24N, 125W) to (49N, 66W)

    # Brazil: (latitude range: 5N to 34S, longitude range: 74W to 35W)
    (5, -74, -34, -35),     # Box 2: from (5N, 74W) to (34S, 35W)

    # Australia: (latitude range: 10S to 44S, longitude range: 113E to 154E)
    (-10, 113, -44, 154),   # Box 3: from (10S, 113E) to (44S, 154E)

    # Canada: (latitude range: 49N to 83N, longitude range: 141W to 52W)
    (49, -141, 83, -52),    # Box 4: from (49N, 141W) to (83N, 52W)

    # Argentina: (latitude range: 22S to 55S, longitude range: 73W to 53W)
    (-22, -73, -55, -53),   # Box 5: from (22S, 73W) to (55S, 53W)

    # China: (latitude range: 18N to 53N, longitude range: 74E to 135E)
    (18, 74, 53, 135),      # Box 6: from (18N, 74E) to (53N, 135E)

    # Russia: (latitude range: 41N to 82N, longitude range: 30E to 180E)
    (41, 30, 82, 180),      # Box 7: from (41N, 30E) to (82N, 180E)

    # South Africa: (latitude range: 22S to 35S, longitude range: 16E to 33E)
    (-22, 16, -35, 33),     # Box 8: from (22S, 16E) to (35S, 33E)

    # India: (latitude range: 8N to 37N, longitude range: 68E to 97E)
    (8, 68, 37, 97),        # Box 9: from (8N, 68E) to (37N, 97E)

    # Greenland: (latitude range: 60N to 84N, longitude range: 20W to 75W)
    (60, -75, 84, -20)      # Box 10: from (60N, 75W) to (84N, 20W)
]
```


# Model Training 

A random forest regressor was trained on data with normalized features and median missing value imputation. The same training and hyperparameters were used for both benchmarks.
```python
# Columns to exclude
exclude_columns = ['Longitude', 'Latitude', 'EMAG2v3']

# Target data
y = data['EMAG2v3']

# Select columns not in exclude_columns using boolean indexing
X = data.loc[:, ~data.columns.isin(exclude_columns)]

# Normalize the features
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# Imputate missing values 
X = X.fillna(X.median())
y = y.fillna(y.median())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)

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

  
### 1 Box Benchmark Metrics

| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             |  3698.262507041346      |
| R<sup>2</sup> Score                  | 0.8499673466604034      |
| Root Mean Squared Error (RMSE)       | 60.813341521752825      |
| Mean Absolute Error (MAE)            | 41.11493823797806       |
| Coefficient of Variation of RMSE     | 0.3873404876069588      |

**red line = line of perfect matching scores**

![download](https://github.com/user-attachments/assets/1baf5af3-51dd-45e9-873f-00a91bcbc1aa)

### 10 Box Benchmark Metrics

| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 1570.834555212438       |
| R<sup>2</sup> Score                  | 0.8477694942863158      |
| Root Mean Squared Error (RMSE)       | 39.633755249943675      |
| Mean Absolute Error (MAE)            | 18.503679782716006      |
| Coefficient of Variation of RMSE     | 0.3901672791427854      |

**red line = line of perfect matching scores**
![image](https://github.com/user-attachments/assets/3edd2c76-f660-417d-a6bc-96d2fa0b75bb)



### Interpretation of Benchmark Performance

When we increased the scope of our test/train split there is an expected increase in performance measured by all metrics other than R<sup>2</sup> Score. Despite this, R<sup>2</sup> score decreased indicating less variance explained, which may be a result of greater variance in larger set. Both models underestimate values in the upper percentiles of their distributions. This may be important in a geophysical or a statistical aspect. The larger benchmark has much larger upper values than the 1 box benchmark.

## Testing Outside Training Box Performance
To test the performance of the model on data from outside of the boundary boxes used for training, the 10 box trained model predicted for the 1 box dataset values,. 


| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 27117.272116792607      |
| R<sup>2</sup> Score                  | -0.07991638520158695      |
| Root Mean Squared Error (RMSE)       | 164.6732282940752      |
| Mean Absolute Error (MAE)            |  123.45282760745039      |
| Coefficient of Variation of RMSE     | 1.0391902545740057      |


![download](https://github.com/user-attachments/assets/13fe0dc9-6a08-4fe5-9a02-c4ad338daa7f)

![download](https://github.com/user-attachments/assets/dbded1a9-c5bf-41e1-a1fc-4ce0cd78fc8a)




In this comparison, the model has a negative R<sup>2</sup> Score, which means the model creates more variance than it explains. This may be due to testing outside of the training area and spatial heterogeneity. 

Spatial Heterogeneity = Different regions can have unique environmental characteristics, such as climate, soil type, vegetation, and topography. A model trained on data from one region may not capture the nuances of a different region.

---

# Region/Hole Benchmarks 
After initial 
