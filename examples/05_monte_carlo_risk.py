"""Uncertain inputs anywhere in the graph.

A ``RandomVariable`` can sit upstream of the attribute you care about. Here
an annual maximum river discharge (Gumbel) sets the water level at a
warehouse through a rating curve, a levee height caps it, and the damage is
a noisy function of the flood depth. The compiled model marginalises the
upstream randomness by Monte Carlo and reports the mean, quantiles, the tail
(CVaR) and exceedance probabilities, each with its sampling error.
"""

import jax
import numpy as np

import prophys as prp
import _style as st

discharge = prp.RandomVariable("discharge", prp.Gumbel(loc=900.0, scale=260.0))  # m3/s
level = 0.9 * (prp.relu(discharge) / 100.0) ** 0.6  # rating curve, stage in m
levee = prp.Param("levee", init=4.5)  # m
depth = prp.softplus(4.0 * (level - levee)) / 4.0  # flood depth behind the levee, m
damage_mean = 2.0 * depth + 0.6 * depth**2  # million EUR

damage = prp.UncertainAttribute(
    "damage", prp.Gaussian(mean=damage_mean, sigma=0.05 + 0.2 * damage_mean), domain="damage in million EUR"
)
compiled = prp.ProbabilityModel(damage).compile(n_samples=20_000, seed=1)

print(compiled.assess("damage"))
print(compiled.assess("damage", "quantile", level=0.99))
print(compiled.assess("damage", "cvar", level=0.99))
p_small = float(compiled.prob("damage", 0.5))
print(f"P(damage > 0.5 MEUR) = {1 - p_small:.3%}")

# Raising the levee is a different parameter value, no rebuilding needed.
for h in (4.5, 5.0, 5.5):
    e = float(compiled.expectation("damage", params={"levee": h}))
    q = float(compiled.quantile("damage", 0.99, params={"levee": h}))
    print(f"levee {h:.1f} m   expected damage {e:.3f} MEUR   99% quantile {q:.2f} MEUR")


def draw(th):
    # Figure, the sampled annual damage and its exceedance curve per levee height.
    draws = np.asarray(compiled.sample("damage", jax.random.PRNGKey(2), (20_000,)))
    draws = draws[0] if draws.ndim > 1 else draws
    q99 = float(compiled.quantile("damage", 0.99))
    cv99 = float(compiled.cvar("damage", 0.99))

    fig, (ax, bx) = st.figure(1, 2)
    flooded = draws[draws > 0.05]
    ax.hist(flooded, bins=60, color=th.series[0], alpha=0.35)
    ax.axvline(q99, color=th.series[1], lw=1.6, label=f"99% quantile {q99:.1f}")
    ax.axvline(cv99, color=th.brand, lw=1.6, ls="--", label=f"CVaR 99% {cv99:.1f}")
    ax.set(xlabel="annual damage [MEUR]", ylabel="years in 20 000", title="Damage in years with a flood", yscale="log")
    ax.legend()

    levels = np.linspace(0.25, 12.0, 48)
    for i, h in enumerate((4.5, 5.0, 5.5)):
        cdf = np.array([float(compiled.prob("damage", float(x), params={"levee": h})) for x in levels])
        exceed = np.where(1 - cdf > 0, 1 - cdf, np.nan)
        bx.semilogy(levels, exceed, color=th.series[i], label=f"levee {h} m")
    bx.set(xlabel="damage [MEUR]", ylabel="annual exceedance probability", title="Exceedance curve")
    bx.legend()
    return fig


st.save("05_monte_carlo_risk", draw)
