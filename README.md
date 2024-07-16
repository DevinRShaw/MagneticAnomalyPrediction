# Magnetic Anomaly Prediction

## Overview

This project aims to predict magnetic anomaly grids using geophysical predictor grids. This is an important component of the MagNav project, which develops military-grade navigation algorithms via magnetic navigation. Due to the lack of quality magnetic data, we leverage machine learning to enhance the accuracy of magnetic anomaly predictions. 

This project builds on the methods of a research paper on prediction of in-situ heat flow {insert reference here}. Data from said paper was used as well as the feature ranking and modeling approaches. A random forest regressor is the inital model used to assess the performance of machine learning in the domain of geophysics. Random forest is simple in applcation, resistant to overfitting and provides model explainability, such as built in feature importance analysis. 

## Table of Contents

- [Data](#data)
- [Feature Ranking](#ranking)
- [Benchmark Datasets](#benchmarks)
- [Model Evaluation](#evaluation)
- [Results](#results)
- [License](#license)


## Data
* The majority of geophysical predictor data is sourced from paper on in-situ heatflow prediction {insert reference here}.

* Other predictors are sourced from geological agencies and then adjusted to match the formatting of research paper data, which is in .nc file format at 100 km<sup>2</sup> resolution {add details about equal area grid}.
*    Notebooks used for standardizing datasets outside of the followed research paper are in the ```feature_creation/file_conversion/``` folder.

  
  Additional information on data and sources is in ```data/metadata.MD```.

![MagneticAnomalyPredictionDataSourcesDiagram drawio](https://github.com/user-attachments/assets/9dadd186-dd41-44b4-924c-2f351adc67c3)




## Ranking 
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. F-score is a measure of how well the predictor explains the variance in the model.

![image](https://github.com/user-attachments/assets/f344e73e-ef2b-4c36-81df-1cf90d0b771e)



{Make sure to fill out with all values this table later}
| Predictor | F-score | 
|----------|----------|
| cm_curie_point_depth | Row 1, Col 2 |
| wgm2012_freeair_ponc | Row 2, Col 2 | 
| gl_elevation | Row 3, Col 2 | 
| rayleigh_group   | Row 1, Col 2 |
| sc_crust_den    | Row 2, Col 2 | 
| interpolated_bouguer    | Row 3, Col 2 |
| igrf_dec                | Row 1, Col 2 |
| love_phase  | Row 2, Col 2 | 
| gl_tot_sed_thick     | Row 3, Col 2 |


## Benchmarks
Selecting for train/test data for chosen features and boundary boxes. A small dataset of size (36297 samples, 1 box) and a large dataset of size (356911 samples, 10 boxes) are created for input to the model. The smaller dataset is used to measure how much sample size improves model performance. The model was not trained on the entire globe mainly due to time to train taking too long for project time constraint.


### Feature Selection 

#### Predictors Trained On
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. The goal of selecting features is to improve model performance without model overfitting. Simply choosing high ranked factors will not yield an optimal result due to learning training data too well to generalize. EMM and MF7 {insert scientific names here} were dropped due to having a much larger prediction power than other predictors, and their close domain relation to the target variable, both of which can cause overfitting. The next {insert the accurate k of kth best predictors that yielded best test}

| Predictors| 
|----------|
| cm_curie_point_depth | 
| wgm2012_freeair_ponc | 
| gl_elevation | 
| rayleigh_group   | 
| sc_crust_den    | 
| interpolated_bouguer    |
| igrf_dec                | 
| love_phase  | 
| gl_tot_sed_thick     | 




### Train/Test Grid Selection 
Grid selection notebook in ```prediction_evaluation/grid_selection.ipynb```. Due to the size of our files and large areas of missing data, it is faster to train/test on boundary boxes. Takes .nc files and creates a CSV file where each (lat,lon) combination is represented in (row = sample) and (column = feature) format.

#### Benchmark CSV Format 
A row = sample, column = feature format allows for train/test data to be moved to the benchmark notebook. ```prediction_evaluation/random_forest_benchmark``` in one package and simplifies input to the model via pandas/Numpy compatibility.

| Latitude       | Longitude      | Target         | Predictor 1     | Predictor 2     | ...  | Predictor n     |
|----------------|----------------|----------------|-----------------|-----------------|------|-----------------|
| Sample 1, Latitude | Sample 1, Longitude | Sample 1, Target | Sample 1, Predictor 1 | Sample 1, Predictor 2 | ...  | Sample 1, Predictor n |
| Sample 2, Latitude | Sample 2, Longitude | Sample 2, Target | Sample 2, Predictor 1 | Sample 2, Predictor 2 | ...  | Sample 2, Predictor n |
| Sample 3, Latitude | Sample 3, Longitude | Sample 3, Target | Sample 3, Predictor 1 | Sample 3, Predictor 2 | ...  | Sample 3, Predictor n |
| ...            | ...            | ...            | ...             | ...             | ...  | ...             |
| Sample n, Latitude | Sample n, Longitude | Sample n, Target | Sample n, Predictor 1 | Sample n, Predictor 2 | ...  | Sample n, Predictor n |


#### Small Benchmark Boundary Box
Both boundary box sets are defined as list of tuples becuase```filter_by_boundary_boxes(df, boundary_boxes)``` using a list of tuples for boundary boxes.
```python
boundary_box = [(-115,33,-83,43)]
```

#### Large Benchmark Boundary Boxes 

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


## Model Training 

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


## Evaluation 

### Metrics 

#### Explanation of Metrics

- **Mean Squared Error (MSE)**: This metric measures the average squared difference between the predicted values and the actual values. It is useful for understanding the overall performance of the model. A lower MSE indicates better model performance.

- **R<sup>2</sup> Score**: Also known as the coefficient of determination, this metric indicates how well the model's predictions fit the actual data. It ranges from 0 to 1, with values closer to 1 indicating a better fit. An R<sup>2</sup> score of 0.8477694942863158 means that approximately 84.77% of the variance in the target variable is explained by the model.

- **Root Mean Squared Error (RMSE)**: This is the square root of the Mean Squared Error and provides a measure of the average magnitude of the prediction errors. RMSE is in the same units as the target variable, making it easier to interpret. A lower RMSE indicates better model performance.

- **Mean Absolute Error (MAE)**: This metric measures the average absolute difference between the predicted values and the actual values. Like RMSE, MAE provides a measure of the average magnitude of the errors, but it is less sensitive to large errors. A lower MAE indicates better model performance.

- **Coefficient of Variation of RMSE (CVRMSE)**: This metric is the RMSE divided by the mean of the observed data, expressed as a percentage. It provides a normalized measure of the prediction error, allowing for comparison across different datasets. A lower CVRMSE indicates better model performance.

  
#### 1 Box Benchmark Metrics

| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             |  3698.262507041346      |
| R<sup>2</sup> Score                  | 0.8499673466604034      |
| Root Mean Squared Error (RMSE)       | 60.813341521752825      |
| Mean Absolute Error (MAE)            | 41.11493823797806       |
| Coefficient of Variation of RMSE     | 0.3873404876069588      |

**red line = line of perfect matching scores**

![download](https://github.com/user-attachments/assets/1baf5af3-51dd-45e9-873f-00a91bcbc1aa)

#### 10 Box Benchmark Metrics

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

When we increased the scope of our test/train split there is an expected increase in performance measured by R<sup>2</sup> Score, which is to be expected in a data science / machine learning setting. Both models underestimate values in the upper percentiles of their distributions. This may be important in a geophysical or a statistical aspect. The larger benchmark has much larger upper values than the 1 box benchmark.

### Testing Outside Training Box Performance
To test the performance of the model on data from outside of the boundary boxes used for training, the 10 box trained model predicted for the 1 box dataset values,. 


| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 1570.834555212438       |
| R<sup>2</sup> Score                  | 0.8477694942863158      |
| Root Mean Squared Error (RMSE)       | 39.633755249943675      |
| Mean Absolute Error (MAE)            | 18.503679782716006      |
| Coefficient of Variation of RMSE     | 0.3901672791427854      |


![download](https://github.com/user-attachments/assets/4367b390-2c45-4ace-b424-afbef582c933)
![download](https://github.com/user-attachments/assets/799b8c34-4592-4203-9aa5-c7345491acfd)

In this comparison, the general "shape" of some areas is captured while other parts are much less accurate than on the in-training box results. This indicates that the spatial aspect of this data is important to prediction. The predictions seem to have a much smaller absolute values on average. More analysis on which datapoints the model performs worse on is needed.

Spatial Heterogeneity = Different regions can have unique environmental characteristics, such as climate, soil type, vegetation, and topography. A model trained on data from one region may not capture the nuances of a different region.


## Results 

Due to the similarity in predictors and domain of research, an initial goal of this project was to exceed the R<sup>2</sup> score acheived by the research paper that we modeled our approach after. 

### In-Situ Heat Flow Prediction Algorithms' Best Scores 
![image](https://github.com/user-attachments/assets/d9b7a0e8-82f4-417e-8978-e3bc17cba7db)


### Magnetic Anomaly Prediction Large Benchmark Scores 



## Future Work


