"""Distributions and fitting them to data.

prophys ships univariate, circular, multivariate and mixture families that
share one interface, ``log_prob``, ``sample``, ``cdf``, ``quantile``,
``mean`` and ``variance``. Their parameters can be ``Param`` leaves, and
``fit_distribution`` estimates them from data by maximum likelihood.
"""

import jax
import numpy as np

import prophys as prp
import _style as st

families = {
    "Weibull(8, 2)": prp.Weibull(scale=8.0, concentration=2.0),
    "LogNormal(1.8, 0.4)": prp.LogNormal(mu=1.8, sigma=0.4),
    "Gamma(4, 0.5)": prp.Gamma(concentration=4.0, rate=0.5),
    "Gumbel(6, 2)": prp.Gumbel(loc=6.0, scale=2.0),
    "Mixture of two Gaussians": prp.Mixture(
        [prp.Gaussian(4.0, 1.0), prp.Gaussian(12.0, 2.0)], logits=[0.0, -0.5]
    ),
}
key = jax.random.PRNGKey(0)
for name, d in families.items():
    draws = np.asarray(d.sample(key, (20_000,)).evaluate())
    print(f"{name:26s} sampled mean {draws.mean():6.2f}   99th percentile {np.quantile(draws, 0.99):6.2f}")

# Synthetic wind speeds, then recover the Weibull parameters from them.
key = jax.random.PRNGKey(3)
truth = prp.Weibull(scale=7.5, concentration=2.2)
data = np.asarray(truth.sample(key, (2000,)).evaluate())

model = prp.Weibull(
    scale=prp.Param("scale", init=4.0, transform="softplus"),
    concentration=prp.Param("k", init=1.0, transform="softplus"),
)
fit = prp.fit_distribution(model, data, n_steps=400, learning_rate=0.05)
print(f"fitted scale {float(fit.params['scale']):.2f} (true 7.5)   shape {float(fit.params['k']):.2f} (true 2.2)")


def draw(th):
    # Figure, the families and the fitted Weibull against a histogram.
    fig, (ax, bx) = st.figure(1, 2)
    x = np.linspace(0.01, 22.0, 500)
    for i, (name, d) in enumerate(families.items()):
        pdf = np.exp(np.asarray(d.log_prob(x).evaluate()))
        ax.plot(x, pdf, color=th.series[i], label=name)
    ax.set(xlabel="value", ylabel="density", title="Five families, one interface")
    ax.legend()

    fitted = prp.Weibull(scale=float(fit.params["scale"]), concentration=float(fit.params["k"]))
    bx.hist(data, bins=40, density=True, color=th.series[0], alpha=0.25, label="2000 samples")
    bx.plot(x, np.exp(np.asarray(fitted.log_prob(x).evaluate())), color=th.brand, label="fitted Weibull")
    bx.set(xlabel="wind speed [m/s]", ylabel="density", title="Maximum likelihood fit")
    bx.legend()
    return fig


st.save("03_distributions", draw)
