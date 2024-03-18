import numpy as np
import scipy
from tqdm import tqdm
import warnings
import time
import matplotlib.pyplot as plt


class ParticleFilter:
    """Marginalised particle filter for NVM an NsigmaM processes observed in Gaussian noise.
    mu_w is incorporated into the state vector, sigma_w is factorised out."""

    def __init__(self, subordinator, N=10 ** 3, sigma_w=None, mu_mu_w=0, Kw=1, Kv=0.1, alpha_w=10**-6, beta_w=10**-6, is_NVM=True, seed=None):
        # Parameters
        self.N = N  # Number of particles
        self.sigma_w = sigma_w  # Sigma_w (None for model with prior, positive real otherwise)
        self.mu_mu_w = mu_mu_w  # Initial mu_w state space mean value
        self.Kw = Kw  # mu_w prior covariance factor (cov = Kw sigma_w^2)
        self.Kv = Kv  # noise covariance factor (cov = Kv sigma_w^2)
        self.subordinator = subordinator  # Subordinator process
        self.alpha_w = alpha_w  # Inverse Gamma sigma_w prior parameter
        self.beta_w = beta_w  # Inverse Gamma sigma_w prior parameter
        self.is_NVM = is_NVM  # Bool for whether model is NVM (True) or NsigmaM (false)

        # Internal variables
        self.omegas = np.zeros(self.N)  # Particle importance weights
        self.unnormalised = np.zeros(self.N)  # Unnormalised importance weights
        self.history = np.zeros((1, self.N))  # History of Kalman means
        self.a_s = np.zeros((2, self.N))  # Particle Kalman means
        self.C_s = np.zeros((2, 2, self.N))  # Particle Kalman covariances (full covariance is sigma_w^2 C)
        self.sum_log_bits = np.zeros(self.N)  # Particle likelihood component
        self.sum_exp_bits = np.zeros(self.N)  # Particle likelihood component
        self.M = None  # Number of time indices
        self.rng = np.random.default_rng(seed=seed)

    def initialise(self):
        """Initialise weights and Kalman parameters."""
        self.omegas = np.ones(self.N) / self.N
        self.a_s[1, :] = np.ones(self.N) * self.mu_mu_w
        self.C_s[1, 1, :] = np.ones(self.N) * self.Kw

    def run(self, times, y_values):
        """Overall method to run particle filter."""
        self.M = len(times)
        self.initialise()
        times = np.insert(times, 0, 0)  # Start from time = 0
        y_values = np.insert(y_values, 0, 0)  # Start from y = 0
        inferred_means = np.zeros(len(times))

        for j in tqdm(range(1, len(times)), colour='WHITE', desc='Timesteps', ncols=150):#, position=1, leave=False):
            # Resample
            inds = self.rng.choice(self.N, size=self.N, p=self.omegas)  # Sample with replacement

            self.a_s = self.a_s[:, inds]  # Kalman means at resampled indices
            self.C_s = self.C_s[:, :, inds]  # Kalman covariances at resampled indices
            self.sum_log_bits = self.sum_log_bits[inds]
            self.sum_exp_bits = self.sum_exp_bits[inds]

            self.history = np.concatenate((self.history, self.a_s[0, :].reshape(1, self.N)), axis=0)
            for i in range(self.N):
                a = np.reshape(self.a_s[:, i], (2, 1))
                C_tilde = self.C_s[:, :, i]

                # Propagate - bootstrap proposal
                jumps = self.subordinator.generate_jumps(time_interval=times[j]-times[j-1])[0]
                dx = np.sum(jumps)
                if self.is_NVM:
                    C_e = dx
                else:
                    C_e = np.sum(jumps**2)

                if j == 1:
                    dx = max(dx, 10 ** -9)  # to ensure C_ii1 is invertible in the first iteration when dx is 0

                # Kalman updates
                Ai = np.array([[1, dx],
                               [0, 1]])
                B = np.array([[1],
                              [0]])
                C = np.array([[1, 0]])

                # (1)
                a_ii1 = Ai @ a
                C_ii1 = Ai @ C_tilde @ Ai.T + C_e * B @ B.T

                # (2)
                # From Cappe, Godsill, Moulines
                K_t = C_ii1 @ C.T / (C @ C_ii1 @ C.T + self.Kv)
                a_ii = a_ii1 + K_t * (y_values[j] - C @ a_ii1)
                C_ii = (np.identity(2) - K_t @ C) @ C_ii1

                # (3)
                y_hat = (C @ a_ii1)[0][0]
                F_i = (C @ C_ii1 @ C.T + self.Kv)[0][0]

                # Exponent:
                exp_bit = - (y_values[j] - y_hat) ** 2 / (2 * F_i)

                # The rest:
                log_bit = - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(F_i)

                # Marginal likelihood
                # log_likelihood(n,t)=sum(log_bit_like(n,:))+alpha_W*log(beta_W)-(alpha_W+t/2)*log(beta_W-
                # sum(exp_bit_like(n,:)))+gammaln(t/2+alpha_W)-gammaln(alpha_W);
                # marginal_likelihood = self.sum_log_bits[i] + log_bit + self.alpha_w * np.log(self.beta_w) - \
                #                       (self.alpha_w + (j+1)/2) * np.log(self.beta_w - self.sum_exp_bits[i] - exp_bit) + \
                #                       scipy.special.loggamma((j+1)/2 + self.alpha_w) - \
                #                       scipy.special.loggamma(self.alpha_w)

                # Incremental likelhood
                # log p(y_n | y_1:n-1) = log p(y_1:n) - log p(y_1:n-1)
                inc_likelihood = log_bit - (self.alpha_w + (j+1)/2)*np.log(self.beta_w - self.sum_exp_bits[i]-exp_bit) \
                                 + (self.alpha_w + j/2)*np.log(self.beta_w - self.sum_exp_bits[i]) + \
                                 scipy.special.loggamma((j+1)/2 + self.alpha_w) - \
                                 scipy.special.loggamma(j/2 + self.alpha_w)

                self.a_s[:, i] = a_ii.flatten()
                self.C_s[:, :, i] = C_ii
                self.sum_log_bits[i] += log_bit
                self.sum_exp_bits[i] += exp_bit

                # Store log weights initially
                if self.sigma_w:
                    # self.omegas[i] = scipy.stats.norm.logpdf(y_values[j], loc=y_hat, scale=self.sigma_w * F_i ** 0.5)
                    self.omegas[i] = log_bit - np.log(self.sigma_w) + exp_bit / self.sigma_w**2
                else:
                    self.omegas[i] = inc_likelihood

            # Normalise
            self.unnormalised = self.omegas
            self.omegas -= np.nanmax(self.omegas)  # Ensure there is no overflow
            self.omegas = np.exp(self.omegas)
            if np.isnan(self.omegas).any():
                warnings.warn('NaN values in omegas')
            self.omegas[np.isnan(self.omegas)] = 0
            self.omegas /= np.sum(self.omegas)

            inferred_means[j] = np.dot(self.omegas, self.a_s[0, :])

        # Update history for final time index
        inds = self.rng.choice(self.N, size=self.N, p=self.omegas)
        self.history = np.concatenate((self.history, self.a_s[0, inds].reshape(1, self.N)), axis=0)
        self.history = self.history[2:, :]  # Remove initial 0's and first iteration
        return inferred_means

    def get_post_parameters(self):
        """Get parameters for calculating posterior distributions of mu_w and sigma_w.
        Returns omegas, alpha_dash's, beta_dash's, m's, c's"""
        alphas_dash = (self.alpha_w + self.M / 2) * np.ones(self.N)
        betas_dash = self.beta_w - self.sum_exp_bits
        ms = self.a_s[1, :]
        cs = self.C_s[1, 1, :]
        if self.sigma_w:
            cs *= self.sigma_w
        return self.omegas, alphas_dash, betas_dash, ms, cs

    def marginal_likelihood(self):
        """Get marginal likelihood for model with and without sigma_w prior.
        Without prior: p(y_1:n | sigma_w^2)
        With prior: p(y_1:n)
        """
        out = 0
        ps = np.exp(self.unnormalised)
        if self.sigma_w:
            for i in range(self.N):
                out += ps[i] * (self.sum_log_bits[i] - self.M * np.log(self.sigma_w)
                                         - self.sum_exp_bits[i] / self.sigma_w**2)
        else:
            for i in range(self.N):
                out += ps[i] * (self.sum_log_bits[i] + self.alpha_w * np.log(self.beta_w)
                                         - (self.alpha_w + self.N/2) * np.log(self.beta_w - self.sum_exp_bits[i])
                                         + scipy.special.loggamma(self.N/2 + self.alpha_w)
                                         - scipy.special.loggamma(self.alpha_w))
        return out
