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

