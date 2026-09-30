"""A first model.

Two houses, two wind turbines and a noise level that falls with distance.
The acceptance drop of the residents is uncertain and follows a Weibull
distribution whose scale depends on the noise. The model is compiled once
and then answers questions about its attribute.
"""

import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Circle

import prophys as prp
import _style as st

site = prp.Frame("site")
houses = prp.PointList(jnp.array([[0.0, 0.0], [200.0, 0.0]]), site)
turbines = prp.PointList(
    prp.Param("pos", shape=(2, 2), init=jnp.array([[80.0, 80.0], [150.0, 20.0]])), site
)

# Distance from each house to its closest turbine, and a simple noise law.
distance = houses.closest_distance(turbines)
level = prp.Param("base", init=100.0) - 20 * prp.log10(distance)

acceptance = prp.UncertainAttribute(
    "acceptance_drop",
    prp.Weibull(scale=1.0 + level, concentration=prp.Param("k", init=2.0)),
)
model = prp.ProbabilityModel(acceptance)
compiled = model.compile()

print("distance to closest turbine [m]", np.round(np.asarray(distance.evaluate()), 1))
print("noise level [dB]               ", np.round(np.asarray(level.evaluate()), 1))
print("expected acceptance drop       ", np.round(np.asarray(compiled.expectation("acceptance_drop")), 2))


def draw(th):
    # Figure, the site plan and the acceptance drop distribution of each house.
    fig, (ax, bx) = st.figure(1, 2)
    t = np.array([[80.0, 80.0], [150.0, 20.0]])
    for x, y in t:
        ax.add_patch(Circle((x, y), 60, color=th.series[1], alpha=0.08, lw=0))
    ax.scatter(*t.T, marker="^", s=120, color=th.brand, label="turbines", zorder=3)
    ax.scatter([0, 200], [0, 0], marker="s", s=90, color=th.series[0], label="houses", zorder=3)
    st.map_axes(ax, th)
    ax.set(xlim=(-40, 240), ylim=(-60, 150), xlabel="x [m]", ylabel="y [m]", title="Site plan")
    ax.legend(loc="upper left")

    scales = 1.0 + np.asarray(level.evaluate())
    x = np.linspace(0.0, scales.max() * 2.2, 400)
    for i, s in enumerate(scales):
        pdf = np.exp(np.asarray(prp.Weibull(scale=float(s), concentration=2.0).log_prob(x).evaluate()))
        bx.fill_between(x, pdf, color=th.series[i], alpha=0.18, lw=0)
        bx.plot(x, pdf, color=th.series[i], label=f"house {i + 1}")
    bx.set(xlabel="acceptance drop", ylabel="density", title="Uncertain attribute per house")
    bx.legend()
    return fig


st.save("01_first_model", draw)
