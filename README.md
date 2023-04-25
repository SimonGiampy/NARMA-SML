## Non-Linear ARMA time series classification with Streaming Machine Learning model classifiers

Author: Simone Giampà, Politecnico di Milano, AY: 2022-2023

University project for the course *Streaming Data Analytics*

Non-linear ARMA time series dataset generation in the notebook:

Streaming Machine Learning models for binary classification of the generated datasets in the notebook:

Characteristics of the datasets:
- no_drift: this dataset presents the non linear ARMA time series without any addition of concept drift
- drift_features: this dataset adds a concept drift in the definition of the features over time, changing the formulas for the single time series and the correlation between the variables
- drift_labels: this dataset adds a concept drift in the definition of the binary labeling function, changing its values and dependency from the features over time


[River](https://riverml.xyz/0.15.0/) machine learning library is used for the streaming machine learning tasks. It provides models and the possibility of training them online. The learning process follows the paradigm for which the model learns from one sample at a time, for the entire duration of the time series, while retaining relatively high classification accuracy during the entire duration of the training.

Metrics employed

temporal augmentation of the models

implementation summary details and expected outcomes, considerations about the concept drift


evaluate:
- temporal correlation in every feature x and label y, through features and labels
- temporal correlation for every feature x and lable y in past data
- feature significativity in prediction
- plots pacf and acf with their analysis and description

SML models:
- HAT, ARF models (with concept drift detectors)
- HAT, ARF models with the addition of temporal augmentation
- kappa statistics
- kappa temporal (in kappa_t)
- models accuracy and the other metrics

temporal augmentation = add past labels in SML classification with different order

apply prequential evaluation with SML models

