import numpy as np
import pickle
import matplotlib.pyplot as plt
from plotting import *
from filter import *


def get_observations(process, noise_sd=0.1, n_observed=200, seed=None, save_data=False, load_data=False):
    """Generate time series and observations for given process."""
    y, t = process.get_time_series()            # 'True' time series
    rng = np.random.default_rng(seed=seed)
    inds = sorted(rng.choice(len(y), size=n_observed, replace=False))
    if load_data:
        with open('y_t.pickle', 'rb') as f:
            y, t = pickle.load(f)
    times = t[inds]                             # Observed time series
    y_s = y[inds] + rng.normal(loc=0, scale=noise_sd, size=n_observed)

    if save_data:
        with open('y_t.pickle', 'wb') as f:
            pickle.dump((y, t), f, pickle.HIGHEST_PROTOCOL)

        plt.plot(t, y)
        plt.show()

    return y, t, y_s, times


def run_particle_filter(subordinator, times, y_s, t=None, y=None, noise_sd=0.1, sigma_w=1,
                        use_prior=True, n_particles=10**3, show_plots=True):
    """Run particle filter.
    times, y_s are observations
    t, y are original time series
    """
    Kv = noise_sd ** 2 / sigma_w ** 2
    if use_prior:
        sigma_w = None

    p_filter = ParticleFilter(subordinator=subordinator, N=n_particles, Kv=Kv, Kw=1, mu_mu_w=0, sigma_w=sigma_w)
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
            print(f'{j + i*len(gammas)} / {len(vs)*len(gammas)}')
            gamma = GammaProcess(gamma=g, v=v)
            ml = run_particle_filter(subordinator=gamma, times=ts, y_s=ys, noise_sd=noise_sd, sigma_w=sigma_w,
                                     use_prior=use_prior, show_plots=False)
            print(f'v = {v}, gamma = {g}, ML = {ml}')
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
