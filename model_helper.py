import math
import numpy as np
import theano.tensor as tt
import pymc3 as pm
import scipy.stats as sc
import theano


def tt_lognormal(x, mu, sigma):
    cdf = tt.exp(pm.Lognormal.dist(mu=mu, sigma=sigma).logcdf(x))
    return cdf[1:] - cdf[:-1]


def tt_cauchy(x, beta, parameter_2):
    cdf = tt.exp(pm.HalfCauchy.dist(beta=beta).logcdf(x))
    return cdf[1:] - cdf[:-1]


def make_delay_matrix(n_rows, n_columns, first_value):
    size = max(n_rows, n_columns)
    mat = np.zeros((size, size))
    for i in range(size):
        diagonal = np.ones(size - i) * (first_value + i)
        mat += np.diag(diagonal, i)
    return mat[:n_rows, :n_columns].astype(int)


def delay_cases(input_arr, parameter_1, parameter_2, distribution, delay_mat):
    probability = distribution(np.arange(delay_mat.shape[0] + 1) - 0.5, parameter_1, parameter_2)
    mat = tt.triu(probability[delay_mat])
    return tt.dot(input_arr, mat)


if __name__ == "__main__":
    cases = [1, 2, 3]
    delay_mat = make_delay_matrix(len(cases), len(cases) + 20)
    a = delay_cases(cases, 3, 1, tt_lognormal, delay_mat)
    print(a.eval().sum())
