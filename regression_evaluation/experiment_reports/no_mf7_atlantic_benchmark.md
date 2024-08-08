
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
    '2_interpolated_emm_',      # Interpolated magnetic field data from the EMM model.   
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
| Mean Squared Error (MSE)             | 366.4861180835103     |
| R<sup>2</sup> Score                  | 0.8536052147179061     |
| Root Mean Squared Error (RMSE)       | 19.143827153511136      |
| Mean Absolute Error (MAE)            | 14.464278765632057      |
| Coefficient of Variation of RMSE     | 0.38261571489170937      |


![download](https://github.com/user-attachments/assets/a31e4a3b-9387-48d1-9617-c6b9f35537be)



### Test Areas Performance


| Metric                               | Value                   |
|--------------------------------------|-------------------------|
| Mean Squared Error (MSE)             | 976.0179797734351      |
| R<sup>2</sup> Score                  | 0.7141128605555672      |
| Root Mean Squared Error (RMSE)       | 31.24128646156293      |
| Mean Absolute Error (MAE)            |  24.462292095761867      |
| Coefficient of Variation of RMSE     | 0.5346841492361942     |


![download](https://github.com/user-attachments/assets/7599cb33-df1c-47c6-840e-1af5444a8cb4)




| Predicted               | Actual                |
|-----------------------|-----------------------|
| ![download](https://github.com/user-attachments/assets/af2d26db-e7b5-4280-875c-061b945c0ac9) | ![download](https://github.com/user-attachments/assets/580709f5-2794-4801-9595-70290aad8f48)|
| ![download](https://github.com/user-attachments/assets/0c1ad277-06d9-4bc3-a707-2799febabaf3) | ![download](https://github.com/user-attachments/assets/402894a2-817e-44a5-89c8-fdbb61378cd8)|
|![download](https://github.com/user-attachments/assets/f91813a5-6d45-42e3-81b9-e096defa4588) | ![download](https://github.com/user-attachments/assets/8fbfe840-1cdd-489f-a8a8-ed8d892fd13a)|


## Feature Importances 

### Random Forest Importances 
| Feature                          | Importance |
|----------------------------------|------------|
| 4_cm_curie_point_depth_          | 0.3979     |
| 6_interpolated_bouguer_          | 0.1763     |
| 10_sc_crust_vs_                  | 0.0687     |
| Longitude                        | 0.0532     |
| Latitude                         | 0.0522     |
| cluster                          | 0.0496     |
| 2_interpolated_emm_              | 0.0431     |
| 3_gl_tot_sed_thick_              | 0.0430     |
| 9__igrf_dec_                     | 0.0420     |
| 7_sc_crust_vp_                   | 0.0315     |
| 5_gl_elevation_                  | 0.0299     |
| 8_sc_crust_den_                  | 0.0125     |

![download](https://github.com/user-attachments/assets/b8b7c8a0-b6b9-48d7-afc2-325ebd85b8ec)





