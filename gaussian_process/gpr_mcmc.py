import numpy as np
import matplotlib.pyplot as plt

import corner
import emcee
import george

def build_gp(theta, t, y, yerr):
    # log likelihood function for GP parameters
    A, l, s = np.exp(theta)
    k_sqexp = george.kernels.ExpSquaredKernel(l ** 2)  # squared exponential kernel
    kernel = A ** 2 * k_sqexp

    gp = george.GP(kernel, mean=np.mean(y), white_noise=np.log(s ** 2))
    gp.compute(t, yerr)

    return gp

def ln_uniform(val, min_val, max_val):
    # Define the uniform prior and return the log of its value.
    assert max_val > min_val
    return np.log(1 / (max_val - min_val)) if min_val <= val <= max_val else -np.inf

def ln_prior(theta, priors):
    # return the log prior
    ln_priors = []
    for i in range (len(theta)):
        ln_priors_i = ln_uniform(theta[i], priors[i][0], priors[i][1])
        ln_priors.append(ln_priors_i)
    ln_prior_sum = np.sum(ln_priors)
    
    if not np.isfinite(ln_prior_sum):
        return -np.inf
    return ln_prior_sum

def ln_prob(theta, t, y, yerr, priors):
    # return the sum of log prior and log likelihood
    ln_prior_sum = ln_prior(theta, priors)
    gp = build_gp(theta, t, y, yerr)
    ln_likelihood = gp.log_likelihood(y, quiet=True)
    return ln_prior_sum + ln_likelihood

def get_initial_ln_prior(y):
    amp_initial = np.std(y)
    length_initial = 1
    noise_initial = 0.1 * amp_initial
    initial_log = np.log([amp_initial, length_initial, noise_initial])

    # amplitude of the kernel is set to 1% of the data RMS, and the white noise is set to 1% of the data RMS, this
    # ensures that we completely capture the dynamics of the data, both for errors (likely >1% of RMS) and full amplitude
    # of the signal
    amp_prior = np.array([0.1 * amp_initial, 10 * amp_initial])
    length_prior = np.array([0.1 * length_initial, 10 * length_initial])
    noise_prior = np.array([0.1 * noise_initial, 10 * noise_initial])

    # everythin done in log space to ensure that we don't bias too much against small values with a uniform prior
    priors_log = np.log([amp_prior, length_prior, noise_prior])

    return initial_log, priors_log

def run_mcmc(t, y, yerr, priors_log, initial_log, nwalkers=20, nsteps=500):
    # MCMC
    ndim = len(initial_log)
    initialize = np.ones(ndim) * 0.1
    p0 = initial_log + initialize * np.random.randn(nwalkers, ndim)

    # with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, args=(t, y, yerr, priors_log))#, pool=pool)
    sampler.run_mcmc(p0, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
        print("Autocorrelation time: ", tau)
    except:
        print("Autocorrelation time error, try with more steps!")

    return sampler

def plot_walkers(sampler, labels=['$\ln{A}$', '$\ln{\ell}$', '$\ln{\sigma}$']):
    samples = sampler.get_chain()
    ndim = samples.shape[2]
    
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plt.show()

def plot_scatter_matrix(sampler, discard=100, thin=1, labels=['$\ln{A}$', '$\ln{\ell}$', '$\ln{\sigma}$']):
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    log_L = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
    theta_results = np.median(samples, axis=0)

    corner.corner(samples, bins=40, show_titles=True, title_fmt=".5f",
                  truths=theta_results, labels=labels, plot_datapoints=False,
                  quantiles=(0.16, 0.5, 0.84))

    plt.show()

    return samples, log_L

def get_theta(samples_y_log):
    n1_results, theta_results, p1_results = np.exp(np.percentile(samples_y_log, axis=0, q=[16, 50, 84]))
    txt = ['Amplitude of the GP','Length of the GP', 'White noise of the GP']
    for i in range(len(theta_results)):
        theta = theta_results[i]
        theta_plus = (p1_results[i]-theta_results[i])
        theta_minus = (theta_results[i]-n1_results[i])
        print (f'{txt[i]}: {theta:.3f} +{theta_plus:.3f} -{theta_minus:.3f}')

    return theta_results