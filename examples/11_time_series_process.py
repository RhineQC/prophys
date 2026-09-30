"""Time series with a stochastic process.

A process turns a distribution into a path with serial structure. Here the
outdoor temperature over the next 48 hours follows an Ornstein Uhlenbeck
process around a daily cycle. Its parameters are first fitted to a week of
hourly readings, then the process drives a path dependent quantity, the
cooling degree hours above 24 degC, whose distribution a model of
independent hours could never produce.
"""

import jax
import jax.numpy as jnp
import numpy as np

import prophys as prp
import _style as st

HOURS = 48
hours = np.arange(HOURS)
cycle = 22.0 + 5.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)  # daily cycle, degC

# 1. Fit the anomaly process to a week of readings (7 paths of 24 hours).
rng = np.random.default_rng(11)
true_theta, true_sigma = 0.15, 1.2
true_phi = np.exp(-true_theta)
true_step = true_sigma * np.sqrt((1 - np.exp(-2 * true_theta)) / (2 * true_theta))
week = np.zeros((7, 24))
for d in range(7):
    x = 0.0
    for h in range(24):
        x = true_phi * x + true_step * rng.normal()
        week[d, h] = x

theta = prp.Param("theta", init=0.5, transform="softplus")
sigma = prp.Param("sigma", init=0.5, transform="softplus")
anomaly_model = prp.ornstein_uhlenbeck(init=0.0, mean=0.0, theta=theta, sigma=sigma, n_steps=24)
fit = prp.fit_distribution(anomaly_model, week, n_steps=800, learning_rate=0.1)
th, sg = float(fit.params["theta"]), float(fit.params["sigma"])
print(f"fitted reversion rate {th:.3f} per hour (true 0.15), diffusion {sg:.2f} (true 1.2)")

# 2. Use the fitted process as an upstream random variable over 48 hours.
anomaly = prp.RandomVariable("anomaly", prp.ornstein_uhlenbeck(init=1.0, mean=0.0, theta=th, sigma=sg, n_steps=HOURS))
temperature = anomaly + cycle
degree_hours = prp.time_sum(prp.relu(temperature - 24.0))
cooling = prp.UncertainAttribute("cooling", prp.Gaussian(mean=degree_hours, sigma=1.0), unit="K*h")
compiled = prp.ProbabilityModel(cooling).compile(n_samples=4000, seed=5)
print(compiled.assess("cooling"))
print(compiled.assess("cooling", "quantile", level=0.95))


def draw(th):
    # Figure, a fan of sampled paths and the distribution of degree hours.
    paths = np.asarray(anomaly.distribution.sample(jax.random.PRNGKey(1), (400,)).evaluate()) + cycle
    dh = np.asarray(degree_hours.evaluate({"anomaly": jnp.asarray(paths - cycle)}))

    fig, (ax, bx) = st.figure(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    for lo, hi, a in ((5, 95, 0.18), (25, 75, 0.3)):
        ax.fill_between(hours, *np.percentile(paths, [lo, hi], axis=0), color=th.series[0], alpha=a, lw=0,
                        label=f"{lo} to {hi} percent")
    for p in paths[:3]:
        ax.plot(hours, p, color=th.brand, lw=0.9, alpha=0.8)
    ax.plot(hours, cycle, color=th.fg, lw=1.2, ls="--", label="daily cycle")
    ax.axhline(24.0, color=th.series[1], lw=1.0)
    ax.set(xlabel="hour", ylabel="temperature [degC]", title="Sampled temperature paths")
    ax.legend(loc="lower right", ncol=3, fontsize=7.5)

    bx.hist(dh, bins=30, color=th.series[1], alpha=0.5)
    bx.set(xlabel="cooling degree hours above 24 degC", ylabel="paths", title="A path dependent quantity")
    return fig


st.save("11_time_series_process", draw)
