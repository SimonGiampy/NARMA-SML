## Non-Linear ARMA time series classification with Streaming Machine Learning model classifiers

**Author**: Simone Giampà, Politecnico di Milano, AY: 2022-2023

University project for the course *Streaming Data Analytics*

- Non-linear ARMA time series dataset generation in the notebook [NARMA Data Generator](NARMA_Data_Generator.ipynb)

- Streaming Machine Learning models for binary classification of the generated datasets in the notebook: [NARMA Streaming ML](NARMA_Streaming_ML_Evaluation.ipynb)

## Objective of the project:

- create non linear ARMA time series composed of several features, with cross correlation and a binary label
- use streaming machine learning model for applying prequential evaluation to classify the time series
- learning one sample at a time, through the entire data series, while maintaining high classification accuracy
- adapt the models trainings to the concept drift of the change of the binary labeling functions

## Characteristics of the datasets:

- `no_drift`: this dataset presents the non linear ARMA time series without any addition of concept drift
- `drift_features`: this dataset adds a concept drift in the definition of the features over time, changing the formulas for the single time series and the correlation between the variables
- `drift_labels`: this dataset adds a concept drift in the definition of the binary labeling function, changing its values and dependency from the features over time

Libraries needed to execute the notebooks: see [requirements](requirements.md)

> [River](https://riverml.xyz/0.15.0/) machine learning library is used for the streaming machine learning tasks. It provides models and the possibility of training them online. The learning process follows the paradigm for which the model learns from one sample at a time, for the entire duration of the time series, while retaining relatively high classification accuracy during the entire duration of the training. This learning process is called *prequential evaluation*.

## Concept Drift in the datasets:
In the datasets, concept drift was added to test the resiliency of the streaming machine learning models to the changing data distribution. The models with ADWIN concept drift detector were able to perform much better overall, compared to the other models without concept drift detection. 

### Evaluation of the datasets:
- temporal correlation in every feature x and label y, through features and labels
- temporal correlation for every feature x and label y in past data
- feature significativity in the predictions
- PACF and ACF plots with their analysis and description
- features and target auto-correlation and cross-correlation plots

### Streaming Machine Learning models
- HAT (Hoeffding Adaptive Tree) and ARF (Adaptive Random Forest) models (with concept drift detectors)
- HAT and ARF models with the addition of temporal augmentation
- Cohen's Kappa and Kappa Temporal statistics, defined in the [Kappa Temporal script](kappa_t.py)
- Models improved with [Temporal Augmentation](temporally_augmented_classifier.py), providing the past labels as additional features for the online evaluation models
