# Monte-Carlo Markov Chains - Variational Inference

## Description
This projects outlines how to perform probabilistic predictions in the case of a Logistic Regression and Gaussian Process Regression by using a Monte-Carlo Markov Chain method (Metropolis-Hastings) and Black-Box Variational Inference.

## Requirements

To install the requirements, use the following commands (with `python>=3.6` enabled by default):
```bash
pip install matplotlib scipy numpy keras tensorflow jax jaxlib
```

## Project Structure

1. Monte-Carlo Markov Chain 
  a. Logisitic Regression 
    * To estimate the predictive distribution, we use Monte-Carlo sampling. 
    * in [metropolis_hastings_logistic.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/distribution_prediction/metropolis_hastings/metropolis_hastings_logistic.py) To sample from the posterior distribution (closed-form) we use Metropolis Hastings 

<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/metropolis_hastings.png" height="400">


## Remarks

The project was completed during the Spring term 2021 Probabilistc Inference Coursework at Imperial College London by Dr Mark van der Wilk and Dr Luca Grillotti. 

