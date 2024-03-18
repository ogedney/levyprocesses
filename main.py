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
n_observed = 200
n_particles = 10**3

g = 2**0.5                   # Subordinator gamma hyperparameter
v = 2                  # Subordinator v hyperparameter
mu_w = -1                # NVM mu_w parameter
sigma_w = 2             # NVM sigma_w parameter
noise_sd = 0.1         # Noise standard deviation

use_prior = True        # Use sigma_w^2 prior
# ----------------------------------------------------------------------

gamma = GammaProcess(gamma=g, v=v)

nvm = NVMProcess(subordinator=gamma, mu_w=mu_w, sigma_w=sigma_w)

y, t, y_s, times = get_observations(process=nvm, noise_sd=noise_sd, n_observed=n_observed)

marginal_likelihood = run_particle_filter(subordinator=gamma, times=times, y_s=y_s, t=t, y=y, noise_sd=noise_sd,
                                          sigma_w=sigma_w, use_prior=use_prior, n_particles=n_particles,
                                          show_plots=True)

print(f'Marginal likelihood = {round(marginal_likelihood, 1)}')

grid_search(times, y_s, gammas=[1.41], vs=[0.2, 2, 10],
            noise_sd=noise_sd, sigma_w=sigma_w, use_prior=True)
