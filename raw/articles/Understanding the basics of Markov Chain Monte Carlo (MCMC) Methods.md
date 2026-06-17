---
title: "Understanding the basics of Markov Chain Monte Carlo (MCMC) Methods"
source: "https://sarowarahmed.medium.com/understanding-the-basics-of-markov-chain-monte-carlo-mcmc-methods-495c257e9ebc"
author:
  - "[[Sarowar Ahmed]]"
published: 2024-06-23
created: 2026-06-17
description: "What is MCMC methods?"
tags:
  - "clippings"
---
## What is MCMC methods?

MCMC methods are a class of algorithms used to approximate complex probability distributions, especially in Bayesian inference. They work by generating a Markov chain whose stationary distribution matches the desired distribution. Think of it as a smart way to sample from a distribution when direct sampling is challenging.

### Real-Life Example

*Scenario*: Suppose we want to estimate the average height of people in a city, but we only have limited data.

*Question*: How can we use MCMC methods to approximate the distribution of heights and estimate the average height?  
Using MCMC Methods:

▪ Step 1: Define the Model: Assume a probability distribution for heights, such as a normal distribution.  
▪ Step 2: Initialize the Chain: Start with an initial guess for the parameters of the distribution.  
▪ Step 3: Generate Samples: Use MCMC methods to generate samples from the distribution, adjusting the parameters to explore the space efficiently.  
▪ Step 4: Estimate Parameters: Calculate the average height based on the generated samples.

![](https://miro.medium.com/v2/resize:fit:1314/format:webp/1*VzC-dzgxhVZUPX_dNHAZXg.png)

### Scenario

Suppose you are a data scientist working on a project to estimate the average height of adult males in a city. You have collected a small sample of height measurements, but you want to infer the population average and the uncertainty around this estimate.

We’ll use Bayesian inference to estimate the mean height (μ) and the standard deviation (σ) of the population. We’ll assume a normal distribution for the heights and use MCMC to sample from the posterior distribution of the parameters.

### Steps

1. **Define the Model**: Assume the heights are normally distributed with unknown mean μ and standard deviation σ.
2. **Specify Priors**: Set priors for μ and σ. We’ll use a normal prior for μ and a half-normal prior for σ.
3. **Likelihood**: Use the normal likelihood for the observed data.
4. **Posterior**: Use MCMC to sample from the posterior distribution of the parameters.

### Sample Data

Let’s use the following heights (in cm) as our observed data:  
heights = \[170,172,168,171,173,169,174,170,172,173\]

### Python Code Implementation

We’ll use the `pymc3` library in Python, which is well-suited for Bayesian inference using MCMC methods.

```c
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Observed data
heights = np.array([170, 172, 168, 171, 173, 169, 174, 170, 172, 173])

# Define the model
with pm.Model() as model:
    # Priors for unknown model parameters
    mu = pm.Normal('mu', mu=170, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)
    
    # Likelihood (sampling distribution) of observations
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=heights)
    
    # Posterior distribution sampling using MCMC (NUTS is a type of MCMC algorithm)
    trace = pm.sample(2000, return_inferencedata=False)

# Summarize the trace
pm.summary(trace)

# Plot the trace and posterior distributions
pm.traceplot(trace)
plt.show()

# Plot posterior distributions using seaborn
plt.figure(figsize=(10, 5))
sns.histplot(trace['mu'], kde=True, label='Posterior of mu')
sns.histplot(trace['sigma'], kde=True, label='Posterior of sigma')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Posterior Distributions of Parameters')
plt.show()

# Print the mean and standard deviation of the posterior samples
mu_mean = np.mean(trace['mu'])
sigma_mean = np.mean(trace['sigma'])
print(f"Estimated mean height (mu): {mu_mean:.2f} cm")
print(f"Estimated standard deviation (sigma): {sigma_mean:.2f} cm")
```

### Explanation

**Define the Model**: We define a probabilistic model with priors for the mean (`mu`) and standard deviation (`sigma`) of the heights.

## Get Sarowar Ahmed’s stories in your inbox

Join Medium for free to get updates from this writer.

**Likelihood**: The likelihood of observing the given heights is modeled as a normal distribution with parameters `mu` and `sigma`.

- `mu` is assigned a normal prior with a mean of 170 cm and a standard deviation of 10 cm.
- `sigma` is assigned a half-normal prior with a standard deviation of 10 cm.

**Sampling**: We use the `pm.sample` function to draw samples from the posterior distribution using the NUTS (No-U-Turn Sampler) algorithm, a type of MCMC method.

**Summary and Visualization**: We summarize the trace to see the posterior estimates for `mu` and `sigma`. We also plot the trace and the posterior distributions to visualize the results.

**Results**: We calculate and print the estimated mean height and the standard deviation from the posterior samples.

### Conclusion

Using MCMC methods, specifically the NUTS sampler in `pymc3`, we can infer the posterior distributions of the parameters `mu` and `sigma` for the population mean height and its uncertainty.

### Why does these matters?

MCMC methods revolutionize Bayesian statistics by providing a flexible and efficient way to sample from complex probability distributions. They have diverse applications in fields like machine learning, finance, and biology, enabling researchers to tackle challenging inference problems with ease.

If you enjoyed this article, feel free to follow me for more insights and updates.  
[***LinkedIn***](https://www.linkedin.com/in/sarowar-ahmed/) [***GitHub***](https://github.com/sarowarahmed)