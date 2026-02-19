# Marginal likelihood

_We are going to do everything to get $p_{\theta}(x)$_

$$
p_{\theta}(x) = \int_{z}p_{\theta}(z,x)dz
$$

But remember, this is intractable.

So everything above, we are cooked.

---

# Bayesian Route

Bayes' Rule

$$
P_{\theta}(z \mid x) = \frac{P_{\theta}(z)\cdot P_{\theta}(x\mid z)}{P_{\theta}(x)}
$$

But don't forget that $P_{\theta}(x)$ is intractable, thus the Posterior is also intractable. However, we actually have a method to solve for the posterior instead. Let's rearrange so that:

$$
\underbrace{p_{\theta}(x)}_{\text{model evidence}} =  \frac{p_{\theta}(x,z)} {\underbrace {[p_{\theta}(z \mid x)]}_{\text{posterior}}}
$$

## Variational inference (VI)

$$
p_{\theta}(z \mid x) \approx q_{\phi}(z\mid x)
$$

- The parameters are shared across observations (amortized Variational Inference)

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

_For completion, we could've also used Jensen's Inequality_: $f(E[X]) \geq E[f(x)]$ (concave)

Therefore the objective to minimize is:

$$
\mathcal{L}(x) =  -\mathbb{E}_{q_{\phi}}\left[ \log\left( \frac{p_{\theta}(z,x)}{q_{\theta}(z \mid x)} \right) \right]
$$

ELBO behaves as our proxy since $D_{KL_{1}}$ cannot be computed.
