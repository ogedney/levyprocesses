"""Initial script to familiarise myself with the models.
Aims: generate sample paths for truncated normal-Gamma process,
then to build histogram of samples to compare to true density.

Next:
- compare histogram to gamma density
- generate normal-Gamma sample paths
- generate NsigmaM sample paths
"""

from processes import *
from plotting import *
from filter import *
from helpers import *
from tqdm import tqdm
import warnings

# warnings.filterwarnings('ignore')

# ng = NVMProcess(subordinator=GammaProcess(), mu_w=1, sigma_w=1)
# ng = NsigmaMProcess(subordinator=TemperedStableProcess(alpha=0.5, beta=1, C=1), mu_w=1, sigma_w=1)
# ng = GammaProcess()
# plot_n_paths(ng, 10)
# plot_hist_of_n(ng, 10**4)

# plot_hist_of_n(ng, 10**5, bins=100)


# stable = StableProcess()
# plot_n_paths(stable, 10)

# generate_and_save_mixture_samples()
# plot_mixture_samples()
# plot_tail_comparison_nsm_mu_w_0()

# plot_tail_comparison_nvm()
# plot_bound_with_s()

# ----------------------------------------------------------------------
# Control panel
n_observed = 50
n_particles = 10**3

g = 2**0.5                   # Subordinator gamma hyperparameter
v = 2                  # Subordinator v hyperparameter
mu_w = 1                # NVM mu_w parameter
sigma_w = 1             # NVM sigma_w parameter
noise_sd = 0.2         # Noise standard deviation

use_prior = True        # Use sigma_w^2 prior
# ----------------------------------------------------------------------

gamma = GammaProcess(gamma=g, v=v)
# gamma = TemperedStableProcess()
# plot_n_paths(gamma, 10)

# generate_and_save_mixture_samples()
# plot_mixture_samples()
# plot_tail_comparison_nvm_TS()
# plot_bound_with_s_nvm_TS()

nvm = NVMProcess(subordinator=gamma, mu_w=mu_w, sigma_w=sigma_w)

# y, t, y_s, times = get_observations(process=nvm, noise_sd=noise_sd, n_observed=n_observed, load_data=True)

# marginal_likelihood = run_particle_filter(subordinator=gamma, times=times, y_s=y_s, t=t, y=y, noise_sd=noise_sd,
#                                           sigma_w=sigma_w, use_prior=use_prior, n_particles=n_particles,
#                                           show_plots=True, nvm=True)
#
# print(f'Marginal likelihood = {round(marginal_likelihood, 1)}')

# grid_search(times, y_s, gammas=[0.01, 0.14, 1.41, 10, 100], vs=[0.02, 0.2, 2, 10, 100],
#             noise_sd=noise_sd, sigma_w=sigma_w, use_prior=True)

# with open('PMCMC_in.pickle', 'wb') as f:
#     pickle.dump((times, y_s), f, pickle.HIGHEST_PROTOCOL)

with open('PMCMC_in.pickle', 'rb') as f:
    times, y_s = pickle.load(f)

# gammas = [2**0.5]
# vs = [2]
# Kvs = [0.04]
# with open('PMCMC_out.pickle', 'wb') as f:
#     pickle.dump((gammas, vs, Kvs), f, pickle.HIGHEST_PROTOCOL)

with open('PMCMC_out.pickle', 'rb') as f:
    gammas, vs, Kvs = pickle.load(f)

bins = np.logspace(-1, 1, 20)
plt.hist(gammas, bins=bins, density=True)
plt.plot(g*np.ones(100), np.linspace(0, 0.6, 100))
plt.xscale('log')
plt.xlabel('$\gamma$')
plt.ylabel('Histogram density')
plt.show()

bins = np.logspace(-1, 2, 20)
plt.hist(vs, bins=bins, density=True)
plt.plot(v*np.ones(100), np.linspace(0, 0.2, 100))
plt.xscale('log')
plt.xlabel('$\\nu$')
plt.ylabel('Histogram density')
plt.show()

bins = np.logspace(-3, 0, 20)
plt.hist(Kvs, bins=bins, density=True)
plt.plot(0.04*np.ones(100), np.linspace(0, 20, 100))
plt.xscale('log')
plt.xlabel('$\kappa_v$')
plt.ylabel('Histogram density')
plt.show()

# gs, vs, Kvs = PMCMC(times, y_s)
