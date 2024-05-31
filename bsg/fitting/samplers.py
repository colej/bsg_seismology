import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Define the model
def run_model(x, y, colored_noise='Harvey', draws=1000, tune=1000, 
              chains=4, cores=4, target_accept=0.95, init='auto', 
              progressbar=True, random_seed=42, num_samples=300):


    idxl = np.where(x<=1.5)[0]
    idxu = np.where(x>1.5)[0]

    xl = x[idxl]
    xu = x[idxu]
    x = np.hstack([xl, xu[::20]])

    yl = y[idxl]
    yu = y[idxu]
    y = np.hstack([yl, yu[::20]])

    plt.loglog(x, y, 'k--')
    plt.show()
    # x = np.array(x)[::10]
    # y = np.array(y)[::10]
    print(max(y), np.median(y), min(y))
    xp = np.linspace(x.min(), x.max(), 1000)
    with pm.Model(coords={'x':x, 'xp':xp}) as model:
        # Prior distributions for he parameters

        # Constant offset representing the instrumental, frequency-independent noise
        # Bounded uniform distribution
        # white_noise = pm.Uniform('white_noise', lower=0.1*np.median(y[-100:]), upper=2.*np.median(y[-100:]))
        white_noise = pm.Truncated('white_noise', pm.Normal.dist(mu=0.9*np.median(y[-100:]), sigma=0.2*np.median(y[-100:])), lower=0, upper=max(y))

        # symmetric Gaussian component
        # amplitude = pm.LogNormal('amplitude', mu=0.8*max(y), sigma=0.3*max(y))
        # amplitude = pm.Uniform('amplitude', lower=50, upper=1.2*max(y))
        amplitude = pm.Normal('amplitude', mu=0.9 * max(y), sigma=0.2*max(y))
        mean = pm.Truncated('mean', pm.LogNormal.dist(mu=0, sigma=1.2), lower=0., upper=1)
        # mean = pm.LogNormal('mean', mu=0, sigma=1.2)
        # mean = pm.Uniform('mean', lower=0.01, upper=2)
        sigma = pm.HalfNormal('sigma', sigma=0.5)
        # gaussian = pm.Deterministic('gaussian', amplitude * pm.math.exp(-1*(x[:,None] - mean)**2 / (2 * sigma**2)))

        if colored_noise == 'Harvey':
            # Harvey profile component
            # alpha = pm.Uniform('alpha', lower=1, upper=max(y))
            alpha1 = pm.Truncated('alpha1', pm.Normal.dist(mu=np.median(x[x<0.1]), sigma=0.2*np.median(x[x<0.1])), lower=0)
            alpha2 = pm.Truncated('alpha2', pm.Normal.dist(mu=np.median(x[x>0.5]), sigma=0.2*np.median(x[x>0.5])), lower=0)
            # beta = pm.LogNormal('beta', mu=1, sigma=1)
            beta1 = pm.Truncated('beta1', pm.LogNormal.dist(mu=1, sigma=1), lower=0, upper=2.5)
            beta2 = pm.Truncated('beta2', pm.LogNormal.dist(mu=1, sigma=1), lower=0, upper=2.5)
            # noise_profile = pm.Deterministic('noise_profile', (2.*np.pi * alpha / beta) / (1 + (x[:,None] / beta)**2))

        elif colored_noise == 'PowerLaw':
            # Power-law component
            alpha = pm.Uniform('alpha', lower=1, upper=max(y))
            beta = pm.LogNormal('beta', mu=1, sigma=1)
            # noise_profile = pm.Deterministic('noise_profile', alpha / (1 + (x[:,None] / beta)**2))

        else:
            raise ValueError('Invalid colored noise model. Please choose between "Harvey" and "PowerLaw"')

        def get_full_profile(x_, name=''):
            
            gaussian = pm.Deterministic('gaussian_' + name, amplitude * pm.math.exp(-0.5*(x_ - mean)**2 / ( sigma**2)))
            if colored_noise == 'Harvey':
                noise_profile_1 = pm.Deterministic('noise_profile_1_'+name, (2.*np.pi * alpha1 / beta1) / (1 + (x_ / beta1)**2))
                noise_profile_2 = pm.Deterministic('noise_profile_2_'+name, (2.*np.pi * alpha2 / beta2) / (1 + (x_ / beta2)**2))
                noise_profile = pm.Deterministic('noise_profile_'+name, noise_profile_1 + noise_profile_2)
            elif colored_noise == 'PowerLaw':
                noise_profile = pm.Deterministic('noise_profile_'+name, alpha / (1 + (x_ / beta)**2))
             
            return pm.Deterministic('profile_' + name, noise_profile + white_noise + gaussian) 

        # Sum of components
        model_full = get_full_profile(model.coords['x'], name='')
        model_pred = get_full_profile(model.coords['xp'], name='pred')

        # Likelihood
        likelihood = pm.Normal('likelihood', mu=model_full, observed=y, dims='x')

        trace = pm.sample( draws=draws, tune=tune, chains=chains, cores=cores, 
                           target_accept=target_accept, init=init, progressbar=progressbar, 
                           random_seed=random_seed)
        post = az.extract(trace, num_samples=num_samples, var_names=['pred'], filter_vars='like')
        full_predictive = post['profile_pred']
        gaussian_predictive = post['gaussian_pred']
        noise_1_predictive = post['noise_profile_1_pred']
        noise_2_predictive = post['noise_profile_2_pred']
        models_ = [full_predictive, gaussian_predictive, noise_1_predictive, noise_2_predictive]

    return trace, post, models_, xp


# def run_sampler(x, y):
# # Generate some example data
# x = np.linspace(-10, 10, 100)
# y = 2 * np.exp(-x**2 / 2) + 1 / (1 + x**2) + np.random.normal(0, 0.1, size=len(x))

# # Fit the model to the data
# with model(x, y):
#     trace = pm.sample()

# # Print the posterior distributions of the parameters
# print(pm.summary(trace))