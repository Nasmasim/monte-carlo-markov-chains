import matplotlib.pyplot as plt
import numpy as np

from gaussian_process import GaussianProcess
from kernels.gaussian_linear_kernel import GaussianLinearKernel
from objective_functions.abstract_objective_function import ObjectiveFunction
from objective_functions.sin import LinearSin
from scipy.stats import multivariate_normal


def get_log_upper_proba_distribution_gp(gaussian_process: GaussianProcess,
                                        theta: np.ndarray):
    """
    This functions evaluates log( p_1(theta | X, y) ) where:
     - p_1 = Z * p
     - p is the posterior distribution
     - p_1 is easy to calculate

    There are 2 methods that you might find useful in the class GaussianProcess:
    - get_log_marginal_likelihood
    - get_log_prior_at

    :param gaussian_process
    :param theta: parameters at which we evaluate p_1. In our example, it is a numpy array (row vector)
    of shape (6,). As our linear + gaussian kernel depends on 6 real numbers.
    :return: log( p_1(theta | X, y) )
    """
    log_likelihood = gaussian_process.get_log_marginal_likelihood(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5])

    log_prior = gaussian_process.get_log_prior_at(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5])
    
    log_posterior = np.sum( log_likelihood ) + log_prior
    
    return log_posterior 


def metropolis_hastings_gaussian_process(gp: GaussianProcess,
                                         number_expected_samples: int,
                                         sigma_exploration_mh: float,
                                         number_hyperparameters_gaussian_process: int):
    """
   Performs a Metropolis Hastings procedure.
   This function is a generator. After each step, it should yield a tuple containing the following elements
   (in this order):
   -  is_sample_accepted (type: bool) which indicates if the last sample from the proposal density has been accepted
   -  np.array(list_samples): numpy array of size (S, 6) where S represents the total number of previously accepted
   samples, and 6 is the number of components in theta in this Gaussian Process regression task.
   -  newly_sampled_theta: numpy array of size (number_hyperparameters_gaussian_process,)
   -  u (type: float): last random number used for deciding if the newly_sampled_theta should be accepted or not.


   :param gp: gaussian process. The goal of this method is to simulate a sampling from the posterior of gp
   :param number_expected_samples: Number of samples expected from the Metropolis Hastings procedure
   :param sigma_exploration_mh: Standard deviation of the proposal density.
   We consider that the proposal density corresponds to a multivariate normal distribution, with:
   - mean = null vector
   - covariance matrix = (sigma_proposal_density ** 2) identity matrix
   :param number_hyperparameters_gaussian_process: Number of hyperparameters for the kernel of the gp.
   """

    list_samples = []  # Every newly_sampled_theta  which is accepted should be added to the list of samples

    newly_sampled_theta = None  # Last sampled parameters (from the proposal density q)

    is_sample_accepted = False  # Should be True if and only if the last sample has been accepted

    u = np.random.rand()  # Random number used for deciding if newly_sampled_theta should be accepted or not

    first_theta = np.zeros(number_hyperparameters_gaussian_process)

    last_theta = first_theta
    
    while len(list_samples) < number_expected_samples:

        u = np.random.rand()

        newly_sampled_theta = np.random.multivariate_normal(mean=last_theta, cov=(sigma_exploration_mh ** 2) * np.identity(number_hyperparameters_gaussian_process) )

        lp_t_tau = get_log_upper_proba_distribution_gp(gp, last_theta)
        lp_t_tau_one = get_log_upper_proba_distribution_gp(gp, last_theta)

        accepted = np.min((1.0, np.exp( lp_t_tau_one - lp_t_tau)))

        is_sample_accepted = accepted > u

        if is_sample_accepted:
            list_samples.append(newly_sampled_theta)
            last_theta = newly_sampled_theta      

        yield is_sample_accepted, np.array(list_samples), newly_sampled_theta, u


def calculate_variance_even_mixture_gaussians(list_means, list_variances):
    array_means = np.array(list_means)
    array_var = np.array(list_variances)

    return np.mean(array_var, axis=0) + np.mean(np.power(array_means, 2), axis=0) - np.power(
        np.mean(array_means, axis=0), 2)


def get_estimated_mean_and_std(gp: GaussianProcess, array_samples_parameters, X):
    functions_samples = []
    X = X.reshape((-1, gp.array_dataset.shape[1]))

    for num_samples, sample_gp_parameter in enumerate(array_samples_parameters):
        gp.set_kernel_parameters(*sample_gp_parameter.flatten())
        functions_samples.append(gp.get_sample(X))

        yield gp.get_sample(X)

        if num_samples % 50 == 0:
            print(f'num samples: {num_samples} \r')


def test_metropolis_hastings(objective_function: ObjectiveFunction,
                             gp: GaussianProcess,
                             number_expected_samples: int,
                             sigma_exploration_mh: float,
                             number_hyperparameters_gaussian_process: int):
    boundaries = objective_function.boundaries
    number_dimensions = len(boundaries)

    array_samples_parameters = None
    for is_accepted_sample, array_samples_parameters, *_ in metropolis_hastings_gaussian_process(gp,
                                                                                                 number_expected_samples,
                                                                                                 sigma_exploration_mh,
                                                                                                 number_hyperparameters_gaussian_process=number_hyperparameters_gaussian_process):
        pass

    if number_dimensions == 1:
        xlim, = boundaries
        x_gt = np.linspace(xlim[0], xlim[1], 100)
        xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 200)

        plt.plot(x_gt, objective_function.evaluate_without_noise(x_gt), c='c')

        plt.title(f"Gaussian Process Regression")
        for function_sample in get_estimated_mean_and_std(gp, array_samples_parameters, xx.reshape((-1, 1))):

            plt.plot(xx, function_sample, alpha=0.3, c='C0')
            plt.scatter(gp.array_dataset,
                        gp.array_objective_function_values,
                        c='m',
                        marker="+",
                        zorder=1000,
                        s=(30,))
            plt.pause(0.05)
        plt.show()


if __name__ == '__main__':
    # obj = UnivariateObjectiveFunction()
    np.random.seed(207)
    obj = LinearSin(0.5)

    initial_dataset = obj.get_uniform_dataset(21).reshape((-1, 1))
    evaluations = obj(initial_dataset)

    # we use a linear combination of a gaussian and a linear kernel: k = k_gaussian + k_linear
    # Then, there are 4 parameters to sample from in the posterior distribution
    kernel = GaussianLinearKernel(0., 0., 0., 0., 0., 0.)
    gp = GaussianProcess(kernel, initial_dataset, evaluations)

    test_metropolis_hastings(obj, gp, 100, sigma_exploration_mh=0.4, number_hyperparameters_gaussian_process=6)
