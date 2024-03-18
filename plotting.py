import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pickle
import math
from scipy.stats import norm, invgamma,t
from processes import *


def plot_n_paths(model, n):
    for i in range(n):
        y, t = model.get_time_series()
        plt.plot(t, y)
    # plt.title(f'{n} {model} sample paths')#, wrap=True, pad=6)
    plt.title('10 NσM (tempered stable process subordinator) sample paths')
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.show()


def plot_hist_of_n(model, n, bins=40):
    dt = 0.75
    vals = model.get_n_final_values(n, time_interval=dt)
    plt.hist(vals, bins=np.linspace(0, 2, 20), density=True)
    plt.title(f'Truncated Gamma density at t = {dt} ({n} samples)')
    plt.xlim(0, 2)
    plt.xlabel('x')
    plt.ylabel('Density at t = 1')
    # x = np.linspace(0, 10, 10000)
    # y = model.wiki_lambda ** model.wiki_gamma * np.multiply(x**(model.wiki_gamma-1), np.exp(- model.wiki_lambda*x)) \
    #     / gamma(model.wiki_gamma)
    # plt.plot(x, y)
    plt.show()


def generate_and_save_mixture_samples():
    nvm = NVMProcess(subordinator=GammaProcess(), mu_w=1, sigma_w=1)
    NVM_vals = nvm.get_n_final_values(10 ** 5)
    with open('nvm_vals.pickle', 'wb') as f:
        pickle.dump(NVM_vals, f, pickle.HIGHEST_PROTOCOL)

    # nsm = NsigmaMProcess(subordinator=GammaProcess(), mu_w=0, sigma_w=1)
    # NSM_vals = nsm.get_n_final_values(10 ** 5)
    # with open('nsm_vals.pickle', 'wb') as f:
    #     pickle.dump(NSM_vals, f, pickle.HIGHEST_PROTOCOL)


def plot_mixture_samples():
    with open('nvm_vals.pickle', 'rb') as f:
        nvm_vals = pickle.load(f)

    with open('nsm_vals.pickle', 'rb') as f:
        nsm_vals = pickle.load(f)

    bins = np.linspace(-7.5, 7.5, 150)

    plt.hist(nvm_vals, bins, alpha=0.7, label='NVM', density=True)
    # plt.hist(nsm_vals, bins, alpha=0.7, label='NσM', density=True)
    # plt.legend(loc='upper right')

    # plt.title('Histograms of 10^5 NVM and NσM values at t = 1 (μ_w = 0, σ_w = 1)', wrap=True)

    mu_w = 1
    sigma_w = 1
    lambda_ = 1
    gamma = 2
    M_1 = gamma / lambda_
    M_2 = gamma * 2 / lambda_ ** 2
    x = np.linspace(-3, 20, 10 ** 4)
    plt.plot(x, norm.pdf(x, loc=mu_w * M_1, scale=(mu_w ** 2 * M_2 + sigma_w ** 2 * M_1)**0.5),
             label='Normal survival function')

    plt.xlabel('Final value')
    plt.ylabel('Normalised sample counts')
    # plt.ylim(0, 0.3)
    plt.show()
    

def get_samples_survival_series(final_values):
    x = np.linspace(-12, 20, 300)
    y = np.zeros(len(x))
    for v in final_values:
        y[x < v] += 1
    y /= len(final_values)
    return x, y


def plot_tail_comparison_nvm():
    with open('nvm_vals.pickle', 'rb') as f:
        nvm_vals = pickle.load(f)

    x, y = get_samples_survival_series(nvm_vals)

    plt.plot(x, y, label='Surviving samples')

    s = 0.6
    mu_w = 1
    sigma_w = 1
    lambda_ = 1
    gamma = 2

    C = (1 - (s * mu_w + 0.5 * s ** 2 * sigma_w ** 2) / lambda_) ** -gamma
    x = np.linspace(-3, 20, 10 ** 4)
    y = C * np.exp(-s * x)
    plt.plot(x, y, label='Tail probability bound (s=0.6)')

    M_1 = gamma / lambda_
    M_2 = gamma * 2 / lambda_ ** 2
    plt.plot(x, norm.sf(x, loc=mu_w*M_1, scale=(mu_w**2*M_2+sigma_w**2*M_1)**0.5), label='Normal survival function')

    plt.ylim(10**-5, 1)
    plt.yscale('log')
    # plt.ylim(0, 0.01)
    plt.xlim(5, 20)
    plt.ylabel('Tail probability')
    plt.xlabel('Final value')
    plt.legend()
    # plt.title('Plot of surviving samples against final value for NVM process with gamma process subordinator')
    plt.show()


def plot_bound_with_s_nvm():
    """Functions to help with plotting variation of bound with s"""

    s = np.linspace(0, 1.5, 200)
    mu_w = 0
    sigma_w = 1
    lambda_ = 1
    gamma = 2
    alpha_15 = np.exp(-15*s) * (1 - (mu_w * s + 0.5 * sigma_w ** 2 * s ** 2)/lambda_) ** - gamma
    alpha_20 = np.exp(-5*s) * (1 - (mu_w * s + 0.5 * sigma_w ** 2 * s ** 2)/lambda_) ** - gamma
    alpha_25 = np.exp(-10*s) * (1 - (mu_w * s + 0.5 * sigma_w ** 2 * s ** 2)/lambda_) ** - gamma

    plt.plot(s, alpha_20, label='X = 5')
    plt.plot(s, alpha_25, label='X = 10')
    plt.plot(s, alpha_15, label='X = 15')
    plt.yscale('log')
    plt.xlabel('s')
    plt.ylabel('Tail probability bound')
    plt.legend()
    plt.ylim(10**-7, 1.2)
    plt.show()


def plot_tail_comparison_nsm_mu_w_0():
    """Plot comparison to bound for NσM process with mu_w = 0"""
    with open('nsm_vals.pickle', 'rb') as f:
        nsm_vals = pickle.load(f)

    x, y = get_samples_survival_series(nsm_vals)

    plt.plot(x, y, label='Surviving samples')

    s = 0.5
    mu_w = 0
    sigma_w = 1
    lambda_ = 1
    gamma = 2

    change = 1
    n = 2
    const = 0
    while change > 0.01:
        a_n = (0.5 * s ** 2 * sigma_w ** 2) ** (n / 2) / math.factorial(n / 2)
        M_n = gamma * math.factorial(n-1) / lambda_ ** n
        new = a_n * M_n
        print(new)
        if const == 0:
            change = 1
        else:
            change = new / const
        const += new
        n += 2
    print(const)

    plt.show()


def get_edges(times, y_true):
    xedges = [(times[i] + times[i+1]) / 2 for i in range(len(times)-1)]
    xedges.insert(0, 0)
    xedges.append(1)
    diff = y_true.max() - y_true.min()
    yedges = np.linspace(y_true.min() - 0.1*diff, y_true.max() + 0.1*diff, 40)
    return xedges, yedges


def get_x_and_y(times, history):
    x = np.array([])
    y = np.array([])
    # print(history.shape)
    for i in range(history.shape[0]):
        x = np.concatenate((x, times[i] * np.ones(history.shape[1])))
        y = np.concatenate((y, history[i, :]))
    return x, y


def plot_particles(times, history, observed, t, y_true, log=True):
    xedges, yedges = get_edges(times, y_true)
    x, y = get_x_and_y(times, history)
    # print(xedges)
    # print(yedges)
    # print(x)
    # print(y)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    # print(H)
    if log:
        H = np.log(H)
    X, Y = np.meshgrid(xedges, yedges)
    # print(H)
    plt.pcolormesh(X, Y, H, cmap='Greens')
    plt.plot(t, y_true, label='True process', color='blue')
    plt.scatter(times, observed, label='Observed process', color='orange')
    plt.legend()
    plt.title('Log histogram values')
    plt.show()


def plot_sigma_w_2_posterior(omegas, alphas_dash, betas_dash, n_points=100):
    x = np.linspace(0, 5, n_points)
    y = np.zeros(n_points)
    for i in range(len(omegas)):
        y += omegas[i] * invgamma.pdf(x, a=alphas_dash[i], scale=betas_dash[i])
    # y += invgamma.pdf(x, a=alphas_dash[0], scale=betas_dash[0])
    plt.plot(x, y)
    plt.title(f'mean alpha\' = {np.mean(alphas_dash[0])}, mean beta\' = {np.mean(betas_dash[0])}')
    plt.xlabel('sigma_w^2')
    plt.show()


def plot_sigma_w_2_likelihood(omegas, sum_F_s, E_s, N, n_points=100):
    x = np.linspace(0.001, 5, n_points)
    y = np.zeros(n_points)
    for i in range(len(omegas)):
        y += omegas[i] * (- N * np.log(x) / 2 - 0.5 * sum_F_s[i] - 0.5 * E_s[i] / x)
    plt.plot(x, y)
    plt.title('Log likelhood of sigma_w^2')
    plt.xlabel('sigma_w^2')
    plt.show()


def plot_mu_w_posterior(omegas, m_s, c_s, alphas_dash, betas_dash, n_points=100):
    x = np.linspace(-3, 3, n_points)
    y = np.zeros(n_points)
    for i in range(len(omegas)):
        y += omegas[i] * t.pdf((alphas_dash[i] / (betas_dash[i]*c_s[i]))**0.5 * (x - m_s[i]), df=2*alphas_dash[i])
    plt.plot(x, y)
    plt.title(f'Posterior distribution of mu_w. Mean m = {np.mean(m_s)}')
    plt.xlabel('mu_w')
    plt.show()


def plot_mu_w_posterior_sigma_w_known(omegas, m_s, c_s, n_points=100):
    x = np.linspace(-3, 3, n_points)
    y = np.zeros(n_points)
    for i in range(len(omegas)):
        y += omegas[i] * norm.pdf(x, loc=m_s[i], scale=c_s[i]**0.5)
    plt.plot(x, y)
    plt.title(f'Posterior distribution of mu_w. Mean m = {np.mean(m_s)}')
    plt.xlabel('mu_w')
    plt.show()

