import numpy as np


class JumpProcess:
    """Parent class for simulation of Levy processes"""
    def get_n_final_values(self, n, time_interval=1):
        """Generates n samples of the process at t=1"""
        out = []
        for i in range(n):
            jumps, times = self.generate_jumps()  # time_interval=time_interval)
            out.append(sum(jumps))
            if i % 500 == 0 and i != 0:
                print(f'{i}/{n}')
        return np.array(out)

    def get_time_series(self, resolution=1000):
        """Generates sample time series on [0, 1]"""
        t = np.linspace(0, 1, resolution)
        y = np.zeros(resolution)
        jumps, times = self.generate_jumps()
        for i in range(len(jumps)):
            y[t > times[i]] += jumps[i]
        return y, t

    def generate_jumps(self, eps=10 ** -6, time_interval=1):
        """Generates jumps for Levy process, using the rejection method"""
        gamma_i = 0
        jumps = []
        while True:
            # Epochs of Poisson process
            gamma_i = gamma_i + np.random.exponential(1)
            # Inverse Levy measure method on Q_0
            jumps.append(self.h_func(gamma_i / time_interval))
            # Stop generation once jumps are below epsilon
            if jumps[-1] < eps or len(jumps) > 10 ** 3:
                jumps.pop()
                break
        jumps = np.array(jumps)
        # Acceptance probabilities
        t = self.thinning_func(jumps)
        u = np.random.uniform(0, 1, size=len(jumps))
        thinned_jumps = jumps[u < t]
        times = np.random.uniform(0, time_interval, size=len(thinned_jumps))
        return thinned_jumps, times

    def h_func(self, x):
        """Placeholder parent class method for inverse Levy method on Q_0"""
        return x

    def thinning_func(self, x):
        """Default acceptance probability (dQ/dQ_0) is 1"""
        return np.ones(len(x))


class GammaProcess(JumpProcess):
    """Gamma process"""
    def __init__(self, gamma=2 ** 0.5, v=2):
        self.gamma = gamma
        self.v = v
        self.beta = gamma ** 2 / 2
        self.C = v
        self.wiki_gamma = self.v
        self.wiki_lambda = 0.5 * self.gamma ** 2

    def __repr__(self):
        return 'truncated Gamma process'

    def h_func(self, x):
        """Inverse Levy measure method  on Q_0(dz) = vz^-1 (1 + 1/2 g^2 z)^-1 dz"""
        return 1 / (self.beta * (np.exp(x / self.C) - 1))

    def thinning_func(self, x):
        """dQ/dQ_0"""
        out = []
        for jump in x:
            out.append((1 + self.beta * jump) * np.exp(- self.beta * jump))
        return out


class StableProcess(JumpProcess):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def __repr__(self):
        return 'truncated α-stable process'

    def h_func(self, x):
        """Inverse Levy measure method - no rejection step required"""
        return x ** (-1 / self.alpha)

    def get_time_series(self, resolution=1000):
        """Parent method overridden to include centering term """
        t = np.linspace(0, 1, resolution)
        y = np.zeros(resolution)
        jumps, times = self.generate_jumps()
        if self.alpha > 1:
            sum_of_ks = len(jumps) ** ((self.alpha - 1) / self.alpha) * self.alpha / (self.alpha - 1)
            y -= sum_of_ks * t
        for i in range(len(jumps)):
            y[t > times[i]] += jumps[i]
        return y, t


class TemperedStableProcess(JumpProcess):
    def __init__(self, alpha=0.5, beta=1, C=1):
        """
        Compared to Barndorff-Nielson
        alpha = kappa
        beta = gamma**(1/kappa)/2.0
        C  = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))
        """
        assert (0 < alpha < 1)
        assert (beta >= 0 and C > 0)
        self.alpha = alpha
        self.beta = beta
        self.C = C

    def __repr__(self):
        return 'truncated tempered stable process'

    def h_func(self, x):
        """Inverse Levy measure method on Q_0"""
        return (self.alpha * x / self.C) ** (-1 / self.alpha)

    def thinning_func(self, x):
        """dQ/dQ_0"""
        return np.exp(-self.beta * x)


class NVMProcess(JumpProcess):
    def __init__(self, subordinator, mu_w=0, sigma_w=1):
        self.subordinator = subordinator
        self.mu_w = mu_w
        self.sigma_w = sigma_w

    def __repr__(self):
        return f'NVM process (μ_w = {self.mu_w}, σ_w = {self.sigma_w}) with {self.subordinator} subordinator'

    def generate_jumps(self, eps=10**-6):
        """Generate jumps from subordinator, then multiply by Gaussian RV's"""
        Z, t = self.subordinator.generate_jumps()
        return self.mu_w * Z + self.sigma_w * np.multiply(Z ** 0.5, np.random.normal(size=len(Z))), t


class NsigmaMProcess(JumpProcess):
    def __init__(self, subordinator, mu_w=0, sigma_w=1):
        self.subordinator = subordinator
        self.mu_w = mu_w
        self.sigma_w = sigma_w

    def __repr__(self):
        return f'NσM process (μ_w = {self.mu_w}, σ_w = {self.sigma_w}) with {self.subordinator} subordinator'

    def generate_jumps(self, eps=10**-6):
        """Generate jumps from subordinator, then multiply by Gaussian RV's"""
        Z, t = self.subordinator.generate_jumps()
        return self.mu_w * Z + self.sigma_w * np.multiply(Z, np.random.normal(size=len(Z))), t
