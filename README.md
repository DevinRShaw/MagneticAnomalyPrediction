# Magnetic Anomaly Prediction

## Overview

This project aims to predict magnetic anomaly grids using geophysical predictor grids. This is a crucial component of the MagNav project, which develops military-grade navigation algorithms via magnetic navigation. Due to the lack of quality magnetic data, we leverage machine learning to enhance the accuracy of magnetic anomaly predictions. The end objective of improving magnetic estimations is to improve the performance of the MagNav project.

## Table of Contents

- [Data](#data)
- [Feature Ranking](#ranking)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Next Steps](#Next)
- [License](#license)


## Data
Information on data files and sources is in metadata.txt 


## Ranking 
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. F-score is a measure of how well the predictor explains the variance in the model.
![Uploading image.png…]()


{Make sure to fill out this table later}
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


## Modeling 

### Train/Test Grid Selection 
Grid selection notebook in ```prediction_evaluation/grid_selection.ipynb```. Due to the size of our files and large areas of missing data, it is faster to train/test on boundary boxes. Takes .nc files and creates a CSV file where each (lat,lon) combination is represented in (row = sample) and (column = feature) format.

#### Benchmark Boundary Boxes 

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

#### Functions for Filtering on Boundary Boxes 
```
# Function to check if a point is within any boundary box
def is_within_boundary_box(coords, box):
    min_longitude, min_latitude, max_longitude, max_latitude = box
    if (min_longitude < coords[0] < max_longitude and min_latitude < coords[1] < max_latitude):
      return True

    return False


# Function to check if a point is within any boundary box
def is_within_boundary_boxes(coords, boundary_boxes):
    for box in boundary_boxes:
        min_longitude, min_latitude, max_longitude, max_latitude = box
        if is_within_boundary_box(coords, box):
            return True
    return False

# Function to filter DataFrame based on boundary boxes
def filter_by_boundary_boxes(df, boundary_boxes):
    filtered_df = df[df.swifter.apply(lambda row: is_within_boundary_boxes((row['Longitude'], row['Latitude']), boundary_boxes), axis=1)]
    return filtered_df
```

### Feature Selection 

#### Predictors Trained 
Predictors were selected based on f-score rankings from ```feature_ranking.ipynb```. F-score is a measure of how well the predictor explains the variance in the model. EMM and MF7 {insert scientific names here} were dropped due to having a much larger prediction power than other predictors, and their close domain relation to the target variable, both of which can cause overfitting. The next {insert the accurate description of predictors that yielded best test}

* cm_curie_point_depth   
* wgm2012_freeair_ponc   
* gl_elevation           
* rayleigh_group        
* sc_crust_den         
* interpolated_bouguer   
* igrf_dec             
* love_phase            
* gl_tot_sed_thick     





