# Monte-Carlo Markov Chains - Variational Inference

## Description
This projects outlines how to perform probabilistic predictions in the case of a Logistic Regression and Gaussian Process Regression by using a Monte-Carlo Markov Chain method (Metropolis-Hastings) and Black-Box Variational Inference. Included are also different Kernel implementations as well as Acquisition Function for performing Bayesian Optimisation. 

## Requirements

To install the requirements, use the following commands (with `python>=3.6` enabled by default):
```bash
pip install matplotlib scipy numpy keras tensorflow jax jaxlib
```

## Project Structure

### A. Gaussian Process
#### Kernels 
1. [gaussian_kernel.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/kernels/gaussian_kernel.py) for the squared exponential (Gaussian/ radial-basis-function) kernel: 
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/Screenshot%202021-03-29%20at%2012.05.53.png" width="50%">
</p>
2. [matern_kernel.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/kernels/matern_kernel.py) to use a 1-time differentiable Gaussian Process with 3/2 Matern Kernel: 
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/Screenshot%202021-03-29%20at%2012.08.49.png" width="50%">
</p>
[gaussian_process.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/gaussian_process.py) implements gaussian process specific functions. By minimising the negative log marginal likelihood and measuring log predictive density (LPD), we can also measure the performance of an optimised GP on a test set. 

### B. Bayesian Optimisation

[expected_improvement.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/acquisition_functions/expected_improvement.py) implementation of expected improvement acquisition function used in [bayesian_optimisation.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/bayesian_optimisation.py) to perform Bayesian Optimisation for an arbitrary objective function. 

### C. Monte-Carlo Markov Chain 
#### C1. Logisitic Regression 
[metropolis_hastings_logistic.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/distribution_prediction/metropolis_hastings/metropolis_hastings_logistic.py) to sample from the posterior distribution (closed-form) we use Metropolis Hastings (Monte Carlo Sampling method). 
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/metropolis_hastings.png" width="50%">
</p>

#### C2. Gaussian Process Regression
[metropolis_hastings_gp.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/distribution_prediction/metropolis_hastings/metropolis_hastings_gp.py) to perform several steps of the Metropolis-Hastings algorithm for the GP Regression using the sum of two kernels as an example: 
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/Screenshot%202021-03-29%20at%2010.31.16.png" width="50%">
</p>
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/gaussian_process_regression.png" width="40%">
</p>

### D. Black Box Variational Inference
#### D1. Logistic Regression
Assuming the parameters are sampled from a normal distribution, we maximise the Evidence Lower Bound (ELBO) and perform variational inference in [blackbox_vi_logistics.py](https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/distribution_prediction/blackbox_vi/blackbox_vi_logistics.py)
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/Screenshot%202021-03-29%20at%2011.47.03.png" width="50%">
</p>

#### D2. Gaussian Process Regression
Using the reparametrisation trick, we compute an approximation of the expected lof marginal likelihood: 
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/Screenshot%202021-03-29%20at%2011.54.01.png" width="50%">
</p>
<p align="center">
<img src="https://github.com/Nasmasim/monte-carlo-markov-chains/blob/main/plots/black_box_VI.png" width="50%">
</p>

## Launching a visualisation

You can visualise the results produced by your implementation by launching the python script contained in the corresponding file.

For example, if you want to visualise your predictions based on the Metropolis-Hastings samples,
in the Logistic Regression, you can execute the following command

```bash
python -m distribution_prediction.metropolis_hastings.metropolis_hastings_logistic
```

## Remarks

The project was completed during the Spring term 2021 Probabilistc Inference Coursework at Imperial College London by Dr Mark van der Wilk and Dr Luca Grillotti. 

