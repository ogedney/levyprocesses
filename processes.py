import numpy as np


class JumpProcess:
    def get_n_final_values(self, n, time_interval=1):
        out = []
        for i in range(n):
            jumps, times = self.generate_jumps(time_interval=time_interval)
            out.append(sum(jumps))
            if i % 500 == 0 and i != 0:
                print(f'{i}/{n}')
        return np.array(out)

    def get_time_series(self, resolution=1000):
        t = np.linspace(0, 1, resolution)
        y = np.zeros(resolution)
        jumps, times = self.generate_jumps()
        for i in range(len(jumps)):
            y[t > times[i]] += jumps[i]
        return y, t

    def generate_jumps(self, eps=10 ** -10, time_interval=1):
        gamma_i = 0
        jumps = []
        while True:
            gamma_i = gamma_i + np.random.exponential(1)
            jumps.append(self.h_func(gamma_i / time_interval))
            if jumps[-1] < eps or len(jumps) > 10 ** 3:
                jumps.pop()
                # print(jumps[-1])
                break
            # if len(jumps) % 100 == 0:
            #     print(len(jumps))
        jumps = np.array(jumps)
        t = self.thinning_func(jumps)
        u = np.random.uniform(0, 1, size=len(jumps))
        thinned_jumps = jumps[u < t]
        times = np.random.uniform(0, time_interval, size=len(thinned_jumps))
        return thinned_jumps, times

    def h_func(self, x):
        return x

    def thinning_func(self, x):
        return np.ones(len(x))


class GammaProcess(JumpProcess):
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
        return 1 / (self.beta * (np.exp(x / self.C) - 1))

    def thinning_func(self, x):
        out = []
        for jump in x:
            out.append((1 + self.beta * jump) * np.exp(- self.beta * jump))
        return out


class StableProcess(JumpProcess):
    def __init__(self, alpha=0.5):
        # assert(0 < alpha < 1)
        self.alpha = alpha

    def __repr__(self):
        return 'truncated α-stable process'

    def h_func(self, x):
        return x ** (-1 / self.alpha)

    def get_time_series(self, resolution=1000):
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
    def __init__(self, alpha, beta, C):
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
        return (self.alpha * x / self.C) ** (-1 / self.alpha)

    def thinning_func(self, x):
        return np.exp(-self.beta * x)


class NVMProcess(JumpProcess):
    def __init__(self, subordinator, mu_w=0, sigma_w=1):
        self.subordinator = subordinator
        self.mu_w = mu_w
        self.sigma_w = sigma_w

    def __repr__(self):
        return f'NVM process (μ_w = {self.mu_w}, σ_w = {self.sigma_w}) with {self.subordinator} subordinator'

    def generate_jumps(self, eps=10 ** -10):
        Z, t = self.subordinator.generate_jumps()
        return self.mu_w * Z + self.sigma_w * np.multiply(Z ** 0.5, np.random.normal(size=len(Z))), t


class NsigmaMProcess(JumpProcess):
    def __init__(self, subordinator, mu_w=0, sigma_w=1):
        self.subordinator = subordinator
        self.mu_w = mu_w
        self.sigma_w = sigma_w

    def __repr__(self):
        return f'NσM process (μ_w = {self.mu_w}, σ_w = {self.sigma_w}) with {self.subordinator} subordinator'

    def generate_jumps(self, eps=10 ** -10):
        Z, t = self.subordinator.generate_jumps()
        return self.mu_w * Z + self.sigma_w * np.multiply(Z, np.random.normal(size=len(Z))), t
