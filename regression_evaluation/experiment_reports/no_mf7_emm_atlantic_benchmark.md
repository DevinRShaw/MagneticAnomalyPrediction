# Predictors Trained On
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. EMM and MF7 were removed from selection to assess model performance without a highly important feature.

```python
features = [
    'cluster',                  # Cluster identifier grouping data points with similar properties.
    'Longitude',                # Geographical coordinate specifying the east-west position.
    'Latitude',                 # Geographical coordinate specifying the north-south position.
    '3_gl_tot_sed_thick_',      # Total sediment thickness in the geological layer.
    '10_sc_crust_vs_',          # Shear wave velocity in the crustal layer.
    '4_cm_curie_point_depth_',  # Depth at which magnetic minerals lose their magnetism (Curie point).
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
| Mean Squared Error (MSE)             |  436.9214102981966     |
| R<sup>2</sup> Score                  |  0.8254694710396124     |
| Root Mean Squared Error (RMSE)       | 20.902665148210087      |
| Mean Absolute Error (MAE)            | 15.584267288037358      |
| Coefficient of Variation of RMSE     | 0.4177685112121156      |

![download](https://github.com/user-attachments/assets/f69ca873-9db4-4612-9285-00f00d0bf775)





### Test Areas Performance


| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 1693.085689285493      |
| R<sup>2</sup> Score                  | 0.5040752992516649      |
| Root Mean Squared Error (RMSE)       | 41.14712249095303      |
| Mean Absolute Error (MAE)            |  32.30510494044794      |
| Coefficient of Variation of RMSE     | 0.7042192135609018     |



![download](https://github.com/user-attachments/assets/cac336dd-f5ba-4e50-8675-6cd0c4ccea1d)



| Predicted               | Actual                |
|-----------------------|-----------------------|
|![download](https://github.com/user-attachments/assets/72409abe-bd8d-43fe-8c21-a3baf280f26a)|![download](https://github.com/user-attachments/assets/8f59571c-eb80-4d6d-8c90-7152cb547d23)|
|![download](https://github.com/user-attachments/assets/a9fb8e6e-237b-4d18-a665-b2b4188ae107) | ![download](https://github.com/user-attachments/assets/0e6c1169-3ad6-4f05-8c19-9738ebd2b3c3)|
|![download](https://github.com/user-attachments/assets/bdb723c9-79a6-4d47-8be4-241e09e82942)|![download](https://github.com/user-attachments/assets/0cf6f90f-9d07-4fc8-90ec-bb61c743fb2b)|



## Feature Importances 

### Random Forest Importances 
| Feature                          | Importance |
|----------------------------------|------------|
| 6_interpolated_bouguer_          | 0.2599     |
| cluster                          | 0.1366     |
| Longitude                        | 0.1353     |
| 10_sc_crust_vs_                  | 0.1164     |
| Latitude                         | 0.0814     |
| 9__igrf_dec_                     | 0.0781     |
| 4_cm_curie_point_depth_          | 0.0575     |
| 7_sc_crust_vp_                   | 0.0449     |
| 3_gl_tot_sed_thick_              | 0.0435     |
| 5_gl_elevation_                  | 0.0353     |
| 8_sc_crust_den_                  | 0.0111     |


![download](https://github.com/user-attachments/assets/ce55c099-d6f1-4674-a4bd-2faf0506c966)

