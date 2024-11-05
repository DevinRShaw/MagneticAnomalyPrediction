# Overview

This is a National Oceanic Atmosphereic Association (NOAA) funded project, carried out at the Univeristy of Colorado Cooperative Institute for Research in Environmental Sciences (CIRES) lab. This project aims to regress magnetic anomaly grid values for use in another CIRES project, MagNav, which uses magnetism to navigate without GPS technology.

A major problem in the MagNav project is a lack of quality maps to train their algorithm with. Due to the lack of quality magnetic data and inaccuracies in traditional modeling, leveraging machine learning to enhance the accuracy of magnetic anomaly predictions may be a viable option. The goal is to explore the viability of machine learning for large scale geophysical modeling, and to develop the approach of creating such a model for improving MagNav maps. The idea is to prototype a model and evaluate the potential of machine learning methods for other geophysics applications. Results of random forest also provide oppurtunity to invesigate feature interactions for magnetism, which can contribute to the domain of geophysics. 

This project builds on the methods of a [research paper on prediction of in-situ heat flow](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GC010913). Data from said paper was used as well as the feature ranking and modeling approaches. A random forest regressor is the inital model used to assess the performance of machine learning in the domain of geophysics. Random forest is simple in applcation, resistant to overfitting and provides model explainability, such as built in feature importance analysis. 

# Table of Contents

- [Data Sources and Format](#data-sources-and-format)
- [Feature Ranking](#ranking)
- [Benchmark Datasets](#benchmarks)
- [Model Evaluation](#evaluation)
- [Results](#results)


# Data Sources and Format
* The majority of geophysical predictor data is sourced from a [paper on in-situ heatflow prediction](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GC010913).

* Other predictors are sourced from geological agencies and then adjusted to match the formatting of research paper data, which is in .nc file format at 100 km<sup>2</sup> resolution {add details about equal area grid}.
* Notebooks used for standardizing datasets outside of the followed research paper are in the ```feature_creation/file_conversion/``` folder.

  
  Additional information on data and sources is in ```data/metadata.MD```.

![MagneticAnomalyPredictionDataSourcesDiagram drawio](https://github.com/user-attachments/assets/9dadd186-dd41-44b4-924c-2f351adc67c3)




# Ranking 
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. F-score is a measure of how well the predictor linearly models a relationship to the target.

![download](https://github.com/user-attachments/assets/955f8626-4eeb-4665-819c-adaca74ac168)



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

Results of various benchmarks can be found in ```regression_evaluation/``` as both notebooks and markdown files explaining the experimentaiton process.

---

# Results   

Due to the similarity in predictors and domain of research, an initial goal of this project was to exceed the R<sup>2</sup> score acheived by the research paper that we modeled our approach after. Although we were not able to train on our entire target dataset, the 10 box benchmark performed very well in terms of comparison to the paper. 

## In-Situ Heat Flow Prediction Algorithms' Best Scores 
![image](https://github.com/user-attachments/assets/d9b7a0e8-82f4-417e-8978-e3bc17cba7db)


## Magnetic Anomaly Predictions
Results of various benchmarks can be found in ```regression_evaluation/experiment_reports``` as both notebooks and markdown files explaining the experimentaiton process.

### Fill vs Adjacent Use Case Comparison

The bar plot below illustrates the comparison between the "Fill" and "Adjacent" experiments across four different scenarios: `full`, `no emm`, `no mf7`, and `no emm/mf7`. Each scenario is represented on the x-axis, with two corresponding bars indicating the values for "Fill" (in blue) and "Adjacent" (in orange).

This visualization helps in understanding the variations in "Fill" and "Adjacent" metrics under different conditions, with positive and negative values clearly differentiated by the horizontal line at y=0.


![download](https://github.com/user-attachments/assets/e617a21e-3949-4f36-b500-ac36e13e042d)


### Fill Model vs MF7 Imputation Comparison 

The bar plot below illustrates the comparison between the "Fill" experiments and using MF7 values to fill in missing values in magnetic anomaly data. MF7 is the current method used by the CIRES team to imputate missing values in the EMAG model, which signifies that the filling algorithm created shows promise for use as a replacement technique moving forward. 


![download](https://github.com/user-attachments/assets/ae4de571-0859-43e8-9693-70090d663800)

---






