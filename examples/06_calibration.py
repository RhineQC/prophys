"""Calibrating a physical model to measurements.

A hot component cools towards the room temperature following Newton's law.
Its cooling rate and the sensor noise are unknown. ``calibrate`` fits them by
maximum likelihood, ``parameter_uncertainty`` turns the curvature of the
likelihood into standard errors, and ``profile_likelihood`` checks the
interval of the rate without assuming a quadratic likelihood.
"""

import warnings

import numpy as np

import prophys as prp
import _style as st

times = np.linspace(0.0, 60.0, 25)  # minutes
T_ROOM, T_START = 21.0, 85.0

rate = prp.Param("rate", init=0.02, bounds=(0.001, 0.5))  # per minute
noise = prp.Param("noise", init=2.0, transform="softplus")  # degC
curve = T_ROOM + (T_START - T_ROOM) * prp.exp(-rate * times)

temperature = prp.UncertainAttribute("temperature", prp.Gaussian(mean=curve, sigma=noise), unit="degC")
compiled = prp.ProbabilityModel(temperature).compile()

# Synthetic readings from a true rate of 0.045 per minute and 1.2 degC noise.
rng = np.random.default_rng(4)
truth = T_ROOM + (T_START - T_ROOM) * np.exp(-0.045 * times)
readings = (truth + 1.2 * rng.normal(size=times.size))[None, :]

fit = prp.calibrate(compiled, "temperature", readings, n_steps=1500, learning_rate=0.1)
print(f"fitted rate {float(fit.params['rate']):.4f} per minute, noise {float(fit.params['noise']):.2f} degC")

u = prp.parameter_uncertainty(compiled, {"temperature": readings}, result=fit)
print(u)

# The time to cool below 40 degC, with its uncertainty from the fit.
t40 = prp.propagate(u, lambda p: np.log((T_START - T_ROOM) / (40.0 - T_ROOM)) / p["rate"])
print("time to reach 40 degC [min]", t40)

grid = np.linspace(0.0425, 0.0475, 13)
# Each grid point is a short refit of the noise. Its convergence notes are
# informative only here, so they are silenced for a tidy output.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    profile = prp.profile_likelihood(
        compiled, {"temperature": readings}, "rate", grid, result=fit, n_steps=400, learning_rate=0.05
    )
print("profile likelihood interval of the rate", np.round(profile.interval, 4))


def draw(th):
    # Figure, the data and fitted curve with a band, the loss and the profile.
    fig, axes = st.figure(1, 3, height=2.9)
    ax, bx, cx = axes
    t = np.linspace(0.0, 60.0, 200)
    draws = prp.sample_parameters(u, 400, seed=0)
    curves = T_ROOM + (T_START - T_ROOM) * np.exp(-np.asarray(draws["rate"])[:, None] * t)
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    k = float(fit.params["rate"])
    ax.fill_between(t, lo, hi, color=th.brand, alpha=0.25, lw=0, label="95% band")
    ax.plot(t, T_ROOM + (T_START - T_ROOM) * np.exp(-k * t), color=th.brand, label="fitted curve")
    ax.scatter(times, readings[0], s=14, color=th.series[1], zorder=3, label="readings")
    ax.set(xlabel="time [min]", ylabel="temperature [degC]", title="Fit to 25 readings")
    ax.legend()

    bx.plot(np.asarray(fit.loss_history), color=th.series[0])
    bx.set(xlabel="optimiser step", ylabel="negative log likelihood", title="Training loss", xscale="symlog", yscale="log")

    cx.plot(grid, np.asarray(profile.log_likelihood) - profile.max_log_likelihood, color=th.brand2, marker="o", ms=3)
    for edge in profile.interval:
        cx.axvline(edge, color=th.muted, lw=0.8, ls=":")
    cx.axhline(-1.92, color=th.muted, lw=0.8, ls="--")
    cx.set(xlabel="rate [1/min]", ylabel="log likelihood ratio", title="Profile of the rate")
    return fig


st.save("06_calibration", draw)
