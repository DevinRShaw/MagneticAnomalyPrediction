
# Feature Selection 
## Feature Rankings
![download](https://github.com/user-attachments/assets/81eb3a33-ef8f-4306-9ee1-61ced0643ccc)

## Predictors Trained On
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. EMM was removed from selection to assess model performance without a highly important feature.

```python
features = [
    'cluster',                  # Cluster identifier grouping data points with similar properties.
    'Longitude',                # Geographical coordinate specifying the east-west position.
    'Latitude',                 # Geographical coordinate specifying the north-south position.
    '3_gl_tot_sed_thick_',      # Total sediment thickness in the geological layer.
    '10_sc_crust_vs_',          # Shear wave velocity in the crustal layer.
    '4_cm_curie_point_depth_',  # Depth at which magnetic minerals lose their magnetism (Curie point).
    '1_interpolated_mf7_',      # Interpolated magnetic field data from the MF7 model.     
    '6_interpolated_bouguer_',  # Interpolated Bouguer gravity anomaly data.
    '9__igrf_dec_',             # Declination (angle between magnetic north and true north) from the IGRF model.
    '5_gl_elevation_',          # Elevation of the ground level relative to sea level.
    '7_sc_crust_vp_',           # P-wave velocity in the crustal layer.
    '8_sc_crust_den_'           # Density of the crustal layer.
]
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
| Mean Squared Error (MSE)             | 420.6867822478973      |
| R<sup>2</sup> Score                  | 0.8319544776204528     |
| Root Mean Squared Error (RMSE)       | 20.510650458917613      |
| Mean Absolute Error (MAE)            | 15.134307080376223      |
| Coefficient of Variation of RMSE     | 0.409933558493992      |


![download](https://github.com/user-attachments/assets/2e204649-60e8-49ea-9021-992616d393d4)


### Test Areas Performance


| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 1574.4244508814422      |
| R<sup>2</sup> Score                  | 0.5388325708524839      |
| Root Mean Squared Error (RMSE)       | 39.67901776608693      |
| Mean Absolute Error (MAE)            |  32.091785315773684      |
| Coefficient of Variation of RMSE     | 0.6790930931378378      |


![download](https://github.com/user-attachments/assets/c979926c-1805-46fd-8396-ba02131b7107)



| Predicted               | Actual                |
|-----------------------|-----------------------|
| ![download](https://github.com/user-attachments/assets/56a89443-a849-47f2-943e-7012fe1cd1bb) | ![download](https://github.com/user-attachments/assets/ae01fd66-b2a8-4cae-a1aa-ba45aedff9ca)|
|  ![download](https://github.com/user-attachments/assets/4b70f38f-b8ff-4a9e-a808-0625dd1b2a4a)|![download](https://github.com/user-attachments/assets/27f85407-05ef-4eb9-8415-6f2794f2b11c)|
|![download](https://github.com/user-attachments/assets/0dc1c27c-369d-4e6f-8ffa-ac9e04db80ea)| ![download](https://github.com/user-attachments/assets/e91261d2-0411-46a2-8a5b-25c8ece0c4f7)|


## Feature Importances 

### Random Forest Importances 
| Feature                          | Importance |
|----------------------------------|------------|
| 4_cm_curie_point_depth_          | 0.2174     |
| 6_interpolated_bouguer_          | 0.1781     |
| 10_sc_crust_vs_                  | 0.1309     |
| 9__igrf_dec_                     | 0.0799     |
| cluster                          | 0.0788     |
| Longitude                        | 0.0741     |
| Latitude                         | 0.0679     |
| 1_interpolated_mf7_              | 0.0605     |
| 3_gl_tot_sed_thick_              | 0.0358     |
| 7_sc_crust_vp_                   | 0.0353     |
| 5_gl_elevation_                  | 0.0329     |
| 8_sc_crust_den_                  | 0.0084     |



![download](https://github.com/user-attachments/assets/6a303276-4fc7-4dc7-9763-0d0dfd8550fb)









