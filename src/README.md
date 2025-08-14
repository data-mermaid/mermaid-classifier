## Scripts

- `classify_coralnet_features.py` - This script is used to classify coralnet features using MLP classifier.


## Classify Coralnet Features

- The given script contains the code to classify coralnet features using MLP classifier.
- There are a few ways to define input labels in `pyspacer`. This script opts for defining the training, test and val labels rather than relying on pyspacer's. 
    - If you want to allow pyspacer to define the labels, you can use `ImageLabels` in `TrainClassifierMsg`. 


