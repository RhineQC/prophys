"""Local and global sensitivity.

A floor beam carries an uncertain line load, its steel stiffness varies from
batch to batch and the built span differs slightly from the drawing. The
midspan deflection follows the textbook formula 5 w L^4 / (384 E I).
``local_sensitivity`` gives derivatives and elasticities with respect to the
design parameters, ``sobol_sensitivity`` splits the variance of the
deflection over the random inputs, and ``conditional_expectation`` evaluates
scenarios for a response surface.
"""

import numpy as np

import prophys as prp
import _style as st

load = prp.RandomVariable("load", prp.Gumbel(loc=12.0, scale=2.5))  # kN/m
stiffness = prp.RandomVariable("stiffness", prp.LogNormal(mu=np.log(205.0), sigma=0.06))  # GPa
span = prp.RandomVariable("span", prp.Gaussian(6.0, 0.03))  # m

inertia = prp.Param("inertia", init=8.36e-5, transform="softplus")  # m4, an IPE 300 section
factor = prp.Param("model_factor", init=1.0)

deflection_mm = factor * 5.0 * (load * 1e3) * span**4 / (384.0 * stiffness * 1e9 * inertia) * 1e3
attribute = prp.UncertainAttribute("deflection", prp.Gaussian(mean=deflection_mm, sigma=0.2), unit="mm")
compiled = prp.ProbabilityModel(attribute).compile(n_samples=8000, seed=0)

print(compiled.assess("deflection"))
local = prp.local_sensitivity(compiled, "deflection")
print(local)
sobol = prp.sobol_sensitivity(compiled, "deflection", n_samples=8000)
print(sobol)

# Scenarios, deflection over a grid of loads and stiffnesses at the nominal span.
loads = np.linspace(6.0, 26.0, 21)
stiffs = np.linspace(175.0, 235.0, 21)
L, E = np.meshgrid(loads, stiffs)
surface = np.asarray(
    compiled.conditional_expectation(
        "deflection", {"load": L.ravel(), "stiffness": E.ravel(), "span": np.full(L.size, 6.0)}
    )
).reshape(L.shape)


def draw(th):
    fig = st.blank(7.4, 3.5)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(L, E, surface, cmap=th.brand_map, rstride=1, cstride=1, linewidth=0.2, edgecolor=th.surface, alpha=0.95)
    ax.set_xlabel("load [kN/m]", fontsize=8)
    ax.set_ylabel("E [GPa]", fontsize=8)
    ax.set_zlabel("deflection [mm]", fontsize=8)
    ax.view_init(elev=24, azim=-128)
    st.pane_3d(ax, th)
    ax.set_title("Response surface", loc="left")

    bx = fig.add_subplot(1, 2, 2)
    names = list(sobol.names)
    y = np.arange(len(names))
    bx.barh(y + 0.18, [sobol.first_order[n] for n in names], height=0.34, color=th.series[0], label="first order")
    bx.barh(y - 0.18, [sobol.total_order[n] for n in names], height=0.34, color=th.brand, label="total order")
    bx.set_yticks(y, names)
    bx.set(xlabel="share of variance", title="Sobol indices", xlim=(0, 1))
    bx.legend(loc="center right")
    return fig


st.save("08_sensitivity", draw)
