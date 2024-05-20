Code for 4th year project (Non-Gaussian Levy Process Theory)

Includes code for simulating gamma, tempered stable and stable subordinator processes as well as NVM and N $\sigma$M processes. 

Has marginalised particle filter implementation, with inference for $\mu_w$ and $\sigma_w$ alongside marginal likelihood calculation.

Particle MCMC is implemented to sample subordinator parameters and $\kappa_v$ using particle marginalised Metropolis-Hastings sampler.


filter.py contains the marginalised particle filter implementation

helpers.py contains code for running subordinator parameter grid search, PMCMC and a wrapper function for running the particle filter more easily.

main.py runs code written in the other files

plotting.py contains functions for making relevant graphs

processes.py contains classes for simulating Levy processes

By Olly Gedney (og288)
