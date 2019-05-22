# Kaggle-Titanic
This repo contains an number of scripts and notebooks trying out things on the Titanic dataset on Kaggle.

## [TitanicSexism](https://github.com/garethjns/Kaggle-Titanic/blob/master/titanicsexism-fairness-in-ml.ipynb)
A tongue-in-cheek look at removing the gender bias in the dataset, includes:
 - Using [IBMs AIF360](https://github.com/IBM/AIF360) to evaluate bias in data and models, and trying out reweighing as a method to remove the bias.
 - An object-orientated approach to preprocessing using sklearn Pipelines, with custom transformers.  

See also: https://www.kaggle.com/garethjns/titanicsexism-fairness-in-ml

## [Shortest Titanic kernel](https://github.com/garethjns/Kaggle-Titanic/blob/master/shortest-titanic-kernel.ipynb)
A fork of [this kernel](https://www.kaggle.com/pavlofesenko/shortest-titanic-kernel-0-78468). Attempts to create a reasonably scoring model using the least code possible. A great example of how not to programme.   
See also: https://www.kaggle.com/garethjns/shortest-titanic-kernel-0-78468

## LightGBM
Examples working with [Microsoft's LightGBM](https://github.com/microsoft/LightGBM)

### [lightgbm.ipynb](https://github.com/garethjns/Kaggle-Titanic/blob/master/lightgbm.ipynb)
Introduction to pre-processing and preparing the data to use in LightGBM.   
See also: https://www.kaggle.com/garethjns/microsoft-lightgbm-0-795

### [lightgbm_run.py](https://github.com/garethjns/Kaggle-Titanic/blob/master/lightgbm_run.py)
Script to prepare data, grid search best model parameters, fit a (slightly more) robust ensemble on multiple data splits. Can score about 0.822 (top 3%) with a lucky random seed.    
See also: https://www.kaggle.com/garethjns/microsoft-lightgbm-with-parameter-tuning-0-822

## [3 seconds and 3 features](https://github.com/garethjns/Kaggle-Titanic/blob/master/3_seconds_3_features.py)
A very simple and fast script fitting a logistic regression model with almost no preprocessing. Can score in the top 10% with a lucky random seed, and is a good example of why such a small dataset is terrible for model performance evaluation!  
See also: https://www.kaggle.com/garethjns/3-seconds-and-3-features-top-10 