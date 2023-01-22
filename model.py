import pymc3 as pm
import numpy as np
import theano.tensor as tt
import model_helper as mh

def admissions_model(cases, admissions_obs):
    """
        Parameters
        ----------
        cases : list or array
            Timeseries (day over day) of newly reported cases
        admissions_obs : list or array
            Timeseries (day over day) of new admissions
        Returns
        -------
        : pymc3.Model
            Returns an instance of pymc3 model
    """

    delay_matrix_0 = mh.make_delay_matrix(len(cases), len(cases), 0)

    with pm.Model() as model:
        # priors
        ph = pm.Uniform(name="pH", lower=0, upper=1)
        admissions_beta = pm.Uniform(name="admissions_beta", lower=0.1, upper=20)
        sigma = pm.Uniform(name="sigma", lower=1, upper=100)

        # train model
        new_hospitalized = ph * cases
        admissions = mh.delay_cases(new_hospitalized, admissions_beta, None, mh.tt_cauchy, delay_matrix_0)
        pm.NegativeBinomial(name="admissions", mu=admissions, alpha=sigma, observed=admissions_obs)

    return model


def occupancy_model(cases, occupancy_obs, ph, admissions_beta):
    """
        Parameters
        ----------
        cases : list or array
            Timeseries (day over day) of newly reported cases
        occupancy_obs : list or array
            Timeseries (day over day) of number of beds occupied
        ph: float
            Probability of being admitted to the hospital upon detection
        admissions_beta: float
            Delay between detection and admission
        Returns
        -------
        : pymc3.Model
            Returns an instance of pymc3 model
    """

    delay_matrix_0 = mh.make_delay_matrix(len(cases), len(cases), 0)

    with pm.Model() as model:
        # prior
        discharges_mu = pm.Uniform(name="discharge_mu", lower=0.01, upper=5)
        discharges_sigma = pm.Uniform(name="discharge_sigma", lower=0.01, upper=5)
        sigma = pm.Uniform(name="sigma", lower=1, upper=200)

        # train model
        new_hospitalized = ph * cases
        admissions = mh.delay_cases(new_hospitalized, admissions_beta, None, mh.tt_cauchy, delay_matrix_0)
        discharges = mh.delay_cases(admissions, discharges_mu, discharges_sigma, mh.tt_lognormal, delay_matrix_0)
        predicted = tt.cumsum(admissions) - tt.cumsum(discharges)

        pm.Normal(name="occupancy", mu=predicted, sigma=sigma, observed=occupancy_obs)

    return model
