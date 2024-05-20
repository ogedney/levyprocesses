import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import invgamma, lognorm
from plotting import *
from filter import *


def get_observations(process, noise_sd=0.1, n_observed=200, seed=None, save_data=False, load_data=False):
    """Generate time series and observations for given process."""
    y, t = process.get_time_series()  # 'True' time series
    rng = np.random.default_rng(seed=seed)
    inds = sorted(rng.choice(len(y), size=n_observed, replace=False))
    if load_data:
        with open('data/y_t.pickle', 'rb') as f:
            y, t = pickle.load(f)
    times = t[inds]  # Observed time series
    y_s = y[inds] + rng.normal(loc=0, scale=noise_sd, size=n_observed)

    if save_data:
        with open('data/y_t.pickle', 'wb') as f:
            pickle.dump((y, t), f, pickle.HIGHEST_PROTOCOL)

        plt.plot(t, y)
        plt.show()

    return y, t, y_s, times


def run_particle_filter(subordinator, times, y_s, t=None, y=None, noise_sd=0.1, sigma_w=1, Kv=None,
                        use_prior=True, n_particles=10 ** 3, show_plots=True, show_bar=True, nvm=True):
    """Run particle filter.
    times, y_s are observations
    t, y are original time series
    """
    if not Kv:
        Kv = noise_sd ** 2 / sigma_w ** 2
    if use_prior:
        sigma_w = None

    p_filter = ParticleFilter(subordinator=subordinator, N=n_particles, Kv=Kv, Kw=1, mu_mu_w=0, sigma_w=sigma_w,
                              show_bar=show_bar, is_NVM=nvm)
    out = p_filter.run(times, y_s)

    omegas, alphas_dash, betas_dash, ms, cs = p_filter.get_post_parameters()
    if show_plots:
        if use_prior:
            plot_sigma_w_2_posterior(omegas, alphas_dash, betas_dash)

            plot_mu_w_posterior(omegas, ms, cs, alphas_dash, betas_dash)
        else:
            plot_mu_w_posterior_sigma_w_known(omegas, ms, cs)

        plot_particles(times, p_filter.history, y_s, t, y)

    return p_filter.marginal_likelihood()


def grid_search(ts, ys, gammas, vs, noise_sd=0.1, sigma_w=1, use_prior=True):
    """Run grid search with a gamma subordinator."""
    # pairs = []
    # for g in gammas:
    #     for v in vs:
    #         pairs.append((g, v))
    likelihoods = np.zeros((len(vs), len(gammas)))

    for i, v in enumerate(vs):
        for j, g in enumerate(gammas):
            print(f'{j + i * len(gammas)} / {len(vs) * len(gammas)}')
            gamma = GammaProcess(gamma=g, v=v)
            ml = run_particle_filter(subordinator=gamma, times=ts, y_s=ys, noise_sd=noise_sd, sigma_w=sigma_w,
                                     use_prior=use_prior, show_plots=False, n_particles=10**3)
            print(f'\n v = {v}, gamma = {g}, ML = {ml}')
            likelihoods[i, j] = ml

    # likelihoods = likelihoods.reshape((len(vs), len(gammas)))
    plt.imshow(likelihoods)
    plt.colorbar()
    plt.yticks(np.arange(len(vs)), labels=vs)
    plt.xticks(np.arange(len(gammas)), labels=gammas)
    plt.title('Marginal likelihoods for gamma process subordinator')
    plt.xlabel('Gamma')
    plt.ylabel('v')
    plt.show()

    print('Marginal Likelihoods:')
    print(likelihoods)


def PMCMC(times, y_s, N_samples=1000):
    """Run Particle MCMC to sample from posterior distribution of subordinator parameters.
    Gamma process used here.
    GRW-MH is used in log domain for strictly positive parameters.

    times, y_s are observations"""

    with open('data/PMCMC_out.pickle', 'rb') as f:
        gammas, vs, Kvs = pickle.load(f)

    # Initialise parameters
    gamma_prev = gammas[-1]
    v_prev = vs[-1]
    Kv_prev = Kvs[-1]
    lg_gamma_prev = np.log(gamma_prev)
    lg_v_prev = np.log(v_prev)
    lg_Kv_prev = np.log(Kv_prev)
    subordinator = GammaProcess(gamma=gamma_prev, v=v_prev)
    # gammas = []
    # vs = []
    # Kvs = []
    alpha = 0.4  # Parameters for inverse gamma prior
    beta = 1
    scale_kv = 0.04*2
    s_kv = 0.5*np.log(10)
    step_size = 0.33 * np.log(10)
    N_accepted = 0

    # Initial log ML
    ML_prev = run_particle_filter(subordinator, times, y_s, None, None, None, None, Kv_prev, show_plots=False,
                                  show_bar=False)

    # Initial log prior
    prior_prev = invgamma.logpdf(gamma_prev, alpha, loc=0, scale=beta) + \
                 invgamma.logpdf(v_prev, alpha, loc=0, scale=beta) + \
                 lognorm.logpdf(Kv_prev, scale=scale_kv, s=s_kv)

    for i in tqdm(range(N_samples), colour='WHITE', desc='Timesteps', ncols=150):
        # Proposal
        lg_gamma_prop = lg_gamma_prev + step_size * np.random.randn()
        gamma_prop = np.exp(lg_gamma_prop)
        lg_v_prop = lg_v_prev + step_size * np.random.randn()
        v_prop = np.exp(lg_v_prop)
        lg_Kv_prop = lg_Kv_prev + step_size * np.random.randn()
        Kv_prop = np.exp(lg_Kv_prop)
        subordinator = GammaProcess(gamma=gamma_prop, v=v_prop)

        # Run PF
        ML_prop = run_particle_filter(subordinator, times, y_s, None, None, None, None, Kv_prop, show_plots=False,
                                      show_bar=False)

        # Proposal log prior
        prior_prop = invgamma.logpdf(gamma_prop, alpha, loc=0, scale=beta) + \
                     invgamma.logpdf(v_prop, alpha, loc=0, scale=beta) + \
                     lognorm.logpdf(Kv_prop, scale=scale_kv, s=s_kv)

        # Log acceptance probability
        lg_acc = min(0, ML_prop + prior_prop - ML_prev - prior_prev)

        lg_u = np.log(np.random.rand())

        if lg_u < lg_acc:
            # Accept proposal
            gammas.append(gamma_prop)
            vs.append(v_prop)
            Kvs.append(Kv_prop)
            gamma_prev, v_prev, Kv_prev = gamma_prop, v_prop, Kv_prop
            lg_gamma_prev, lg_v_prev, lg_Kv_prev = lg_gamma_prop, lg_v_prop, lg_Kv_prop
            ML_prev, prior_prev = ML_prop, prior_prop
            N_accepted += 1
        else:
            # Reject proposal
            gammas.append(gamma_prev)
            vs.append(v_prev)
            Kvs.append(Kv_prev)

        if (i+1) % 10 == 0:
            with open('data/PMCMC_out.pickle', 'wb') as f:
                pickle.dump((gammas, vs, Kvs), f, pickle.HIGHEST_PROTOCOL)
            print(f'\n Current run: {i+1}/{N_samples} samples, acceptance percentage = '
                  f'{round(100 * N_accepted / (i+1), 1)}. Total samples: {len(gammas)}')

    return gammas, vs, Kvs
