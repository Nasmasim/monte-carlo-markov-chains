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
* Logisitic Regression 
    * To estimate the predictive distribution, we use Monte-Carlo sampling. 
    * in [metropolis_hastings_logistic.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/distribution_prediction/metropolis_hastings/metropolis_hastings_logistic.py) to sample from the posterior distribution (closed-form) we use Metropolis Hastings 
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/metropolis_hastings.png" width="50%">
</p>
* Gaussian Process Regression
  * We use the sum of two kernels as an example: a gaussian kernel and a linear kernel. 
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/Screenshot%202021-03-29%20at%2010.31.16.png
" width="50%">
</p>
  * in [metropolis_hastings_gp.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/distribution_prediction/metropolis_hastings/metropolis_hastings_gp.py) we perform several steps of the Metropolis-Hastings algorithm for the GP Regression

<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/gaussian_process_regression.png
" width="100%">
</p>

## Remarks

The project was completed during the Spring term 2021 Probabilistc Inference Coursework at Imperial College London by Dr Mark van der Wilk and Dr Luca Grillotti. 

