# Overview

This is a National Oceanic Atmosphereic Association (NOAA) funded project, carried out at the Univeristy of Colorado Cooperative Institute for Research in Environmental Sciences (CIRES) lab. This project aims to regress magnetic anomaly grid values for use in another CIRES project, MagNav, which uses magnetism to navigate without GPS technology.

A major problem in the MagNav project is a lack of quality maps to train their algorithm with. Due to the lack of quality magnetic data and inaccuracies in traditional modeling, leveraging machine learning to enhance the accuracy of magnetic anomaly predictions may be a viable option. The goal is to explore the viability of machine learning for large scale geophysical modeling, and to develop the approach of creating such a model for improving MagNav maps. The idea is to prototype a model and evaluate the potential of machine learning methods for other geophysics applications. Results of random forest also provide oppurtunity to invesigate feature interactions for magnetism, which can contribute to the domain of geophysics. 

This project builds on the methods of a [research paper on prediction of in-situ heat flow](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GC010913). Data from said paper was used as well as the feature ranking and modeling approaches. A random forest regressor is the inital model used to assess the performance of machine learning in the domain of geophysics. Random forest is simple in applcation, resistant to overfitting and provides model explainability, such as built in feature importance analysis. 

# Table of Contents

- [Data Sources and Format](#data-sources-and-format)
- [Feature Ranking](#ranking)
- [Benchmark Datasets](#benchmarks)
- [Model Evaluation](#evaluation)
- [Conclusions](#conclusions)
- [Contributing](#contributing)
- [Future Work](#future-work)


# Data Sources and Format
* The majority of geophysical predictor data is sourced from a [paper on in-situ heatflow prediction](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GC010913).

* Other predictors are sourced from geological agencies and then adjusted to match the formatting of research paper data, which is in .nc file format at 100 km<sup>2</sup> resolution {add details about equal area grid}.
* Notebooks used for standardizing datasets outside of the followed research paper are in the ```feature_creation/file_conversion/``` folder.

  
  Additional information on data and sources is in ```data/metadata.MD```.

![MagneticAnomalyPredictionDataSourcesDiagram drawio](https://github.com/user-attachments/assets/9dadd186-dd41-44b4-924c-2f351adc67c3)




# Ranking 
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. F-score is a measure of how well the predictor linearly models a relationship to the target.

![image](https://github.com/user-attachments/assets/f344e73e-ef2b-4c36-81df-1cf90d0b771e)



| Filename                                     | Value                   | P-value                           |
|----------------------------------------------|-------------------------|-----------------------------------|
| standardized_interpolated_mf7_100km^2.nc     | 1560109.7038581178      | 0.0                               |
| standardized_interpolated_emm_100km^2.nc     | 157150.1145795856       | 0.0                               |
| standardized_gl_tot_sed_thick_100km^2.nc     | 1405.0289900456914      | 1.843757556141502e-307            |
| standardized_cm_curie_point_depth_100km^2.nc | 770.6934776005627       | 1.3027033330072561e-169           |
| standardized_gl_elevation_100km^2.nc         | 765.3689621463677       | 1.8724119603386416e-168           |
| standardized_interpolated_bouguer_100km^2.nc | 438.1094275812039       | 2.814682176817089e-97             |
| standardized_sc_crust_vp_100km^2.nc          | 373.40984527966464      | 3.4066938689985076e-83            |
| standardized_sc_crust_den_100km^2.nc         | 360.99844612470224      | 1.716200855210849e-80             |
| standardized_igrf_dec_100km^2.nc             | 266.3216746121886       | 7.209559708682005e-60             |
| standardized_sc_crust_vs_100km^2.nc          | 241.32763047829047      | 2.0244078772371743e-54            |
| standardized_love_group_100km^2.nc           | 112.26114486580741      | 3.133831036455731e-26             |
| standardized_love_phase_100km^2.nc           | 101.8480245635898       | 5.997566505014905e-24             |
| standardized_rayleigh_group_100km^2.nc       | 75.50813975888067       | 3.639931384364718e-18             |
| standardized_wgm2012_freeair_ponc_100km^2.nc | 55.661609512607626      | 8.609466400441216e-14             |
| standardized_rayleigh_phase_100km^2.nc       | 54.23154043467896       | 1.7822622099189852e-13            |
| standardized_sl_vgg_eot_100km^2.nc           | 47.026006314980975      | 7.005778545104857e-12             |
| standardized_sc_crust_age_100km^2.nc         | 39.76610350941027       | 2.8629047779405667e-10            |
| standardized_igrf_inc_100km^2.nc             | 9.706479949891724       | 0.0018362026258532354             |

---

# Benchmarks

## Small/Large Benchmark
Selecting for train/test data for chosen features and boundary boxes. A small dataset of size (36297 samples, 1 box) and a large dataset of size (356911 samples, 10 boxes) are created for input to the model. The goal is to see if the training size affects accuracy and to measure the performance of the model trained on large dataset somewhere outside of training. 
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



## Region/Holes Benchmark 
After the intial results of the Small/Large Benchmark, we trained the model on a small region (10 x 10 degrees) and tested the model's performance in predicting holes of the map (1 x 1 degrees). This tests if the model can accurately predict values outside of the training set that are spatially close to the training set. If model can accurately learn and then imputate values in an area, there are many possible applicaitons. 



---

# Conclusions  

Due to the similarity in predictors and domain of research, an initial goal of this project was to exceed the R<sup>2</sup> score acheived by the research paper that we modeled our approach after. Although we were not able to train on our entire target dataset, the 10 box benchmark performed very well in terms of comparison to the paper. 

## In-Situ Heat Flow Prediction Algorithms' Best Scores 
![image](https://github.com/user-attachments/assets/d9b7a0e8-82f4-417e-8978-e3bc17cba7db)


## Magnetic Anomaly Prediction Large Benchmark Scores 
![image](https://github.com/user-attachments/assets/d2662a35-fdb7-47cf-9b25-5706f7a3dfe9)


# Contributing
## Using Models 
After training the model, we can save our trained model to a serialized format, pickle, that allows us to use the model in other programs. Model pickles are named respective to their training sets in ```trained_models/```.

### Saving Model to Pickle
```python
#saving to serialized format 
import pickle
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
```
### Loading Model from Pickle 
```python
#loading from serialized format
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
```

## Training Models 
Training for this application is simplified via the ```regression_evaluation/grid_selection.ipynb``` notebook. 
### Process 
* Download the Standardized Data release zip file, which contains all our predictors and target.
* Use files in ```grid_selection.ipynb```
  - Select features to train model with
  - Create list of boundary boxes to sample
* Download .csv benchmark dataset
* Preprocess values for input to model (In benchmark) 
  - Imputate missing values
  - Normalize/Standardize values 
* Input to model in benchmark notebook


# Future Work
There are many potential methods of improving the performance of random forest regression on magnetic anomaly values, given current state of model performance. 

* Spatial Features
  - Include spatial information as features in your model. This can help the model learn spatial dependencies and improve its generalization to new areas.
  - Latitude and Longitude were ranked highly in regional benchmarking





