
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
|  ![download](https://github.com/user-attachments/assets/f944e97d-1363-4f3c-81a7-8da9ab0cc1e1) | ![download](https://github.com/user-attachments/assets/ce702114-c59e-4b1e-954c-a96926a3c3ed) |
|  ![download](https://github.com/user-attachments/assets/b92f125b-968e-47fb-93f6-ef2fd55febe9) |![download](https://github.com/user-attachments/assets/37339c85-1a0c-4f53-bca7-2b0e03d0477a) |
|![download](https://github.com/user-attachments/assets/e73e8024-cb21-4d3a-8a0a-d1d6ae1f3920) | ![download](https://github.com/user-attachments/assets/1e7fea86-570f-4dd0-916d-a442f673ec85) |


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









