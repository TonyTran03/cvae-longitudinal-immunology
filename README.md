# Conditional Variational Autoencoder for Synthetic Longitudinal Immune Data

This repository implements a Conditional Variational Autoencoder (CVAE) for generating synthetic longitudinal immune-response data. The work extends the modeling framework introduced in *"Modelling of longitudinal immune profiles reveals distinct immunogenic signatures following five COVID-19 vaccinations among people living with HIV"*.

The original study demonstrated that high-dimensional immune measurements contain stable immunogenic signatures capable of distinguishing between HIV-positive and HIV-negative cohorts using machine learning classifiers. In addition to evaluating predictive structure, the study explored synthetic data generation through classical statistical and resampling-based approaches.

This project investigates a model-based generative alternative. Rather than reproducing marginal feature behavior alone, the CVAE aims to learn a structured latent representation of immune trajectories while explicitly conditioning on clinically relevant variables such as HIV status, vaccine dose, and visit timing.

The central objective is to generate synthetic participant trajectories that preserve:

- Marginal feature distributions  
- Cross-feature dependence structure  
- Longitudinal relationships across study visits  
- Class-conditional immunological signatures  

By introducing conditional latent variables, the model provides a mechanism for disentangling cohort-level effects from participant-level variability, enabling controlled sampling of virtual immune-response profiles.

---

# Probabilistic Framework

## Conditional Generative Model

The Conditional Variational Autoencoder (CVAE) models the conditional data distribution

$$
p_{\theta}(x \mid c)
$$

where:

- \( x \) denotes observed immune-response measurements  
- \( c \) denotes conditioning variables (e.g., HIV status, vaccine dose, visit index)

# Objective function
#### Kullback-Leibler (KL) Divergence
$$
D_{KL}(Q || P) = \int_{z}Q(z) \log\left( \frac{Q(z)}{P(z)} \right) dz
$$
- Always positive
When we try and solve, we get it to the form:

$$
D_{KL}(Q || P)  = \log p_{\theta}(x) - \mathbb{E}_{q_{\phi}}[\log p_{\theta}(z,x)-\log q_{\phi}]
$$
- $P = p_{\theta}(z \mid x)$ (no typo) (true posterior given by Bayes)
- $Q = q_{\phi}(z \mid x)$ (encoder distribution)
As shuffling gives us:

$$
\begin{align}
\log p_{\theta}(x)  &= \underbrace{D_{KL}(Q || P)}_{ gap }  + \mathbb{E}_{q_{\phi}}[\log p_{\theta}(z,x)-\log q_{\phi}] \\
\implies \log p_{\theta}(x)& \geq \underbrace {\mathbb{E}_{q_{\phi}}[\log p_{\theta}(z,x)-\log q_{\phi}]}_{\text{Evidence lower fbound (ELBO)}}
\end{align}
$$
since
$$
D_{KL}(q_{\phi}(z \mid x) || p_{\theta}(z \mid x)) \geq 0
$$
- gap: How wrong is our approximate posterior vs the true posterior?
- We pretty much ignore this because **we can't compute this**
- Comparing:
	- Approx posterior vs
	- True posterior

*For completion, we could've also used Jensen's Inequality*: $f(E[X]) \geq E[f(x)]$ (concave)

Therefore the objective to minimize is:

$$
\mathcal{L}(x) =  -\mathbb{E}_{q_{\phi}}\left[ \log\left( \frac{p_{\theta}(z,x)}{q_{\theta}(z \mid x)} \right) \right]
$$

ELBO behaves as our proxy since $D_{KL_{1}}$ cannot be computed.
