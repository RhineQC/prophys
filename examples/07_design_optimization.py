"""Design optimisation with constraints.

Three wind turbines must stand inside a permitted area. Their positions are
parameters, and gradient descent moves them to minimise the noise at the
loudest of six houses, while a quadratic penalty keeps them inside the area and at least
150 m apart. The distance queries run in their smooth form so that the
gradient is informative everywhere.
"""

import jax.numpy as jnp
import numpy as np

import prophys as prp
import _style as st

site = prp.Frame("site", units="m")
area_xy = np.array([[150.0, 150.0], [650.0, 120.0], [700.0, 450.0], [420.0, 620.0], [180.0, 500.0]])
area = prp.Polygon(jnp.array(area_xy), site)
houses_xy = np.array([[80.0, 80.0], [380.0, 60.0], [780.0, 300.0], [600.0, 700.0], [90.0, 620.0], [420.0, 380.0]])
houses = prp.PointList(jnp.array(houses_xy), site)

start = np.array([[300.0, 280.0], [500.0, 300.0], [400.0, 470.0]])
pos = prp.Param("pos", shape=(3, 2), init=jnp.array(start))
turbines = prp.PointList(pos, site)


def noise_at(receivers, turbine_points):
    """Sound level [dB] at each receiver, the energy sum of all turbines."""
    d = receivers.pairwise_distance(turbine_points) + 1.0
    level = 104.0 - 20.0 * prp.log10(d) - 0.005 * d
    return 10.0 * prp.log10(prp.sum_(10.0 ** (level / 10.0), axis=1))


# The loudest house counts, through a smooth maximum over the six levels.
objective = prp.softmax(noise_at(houses, turbines), tau=1.0)
inside = area.signed_distance(turbines) + 20.0  # at least 20 m inside the boundary
spacing = turbines.pairwise_distance(turbines) + 1e3 * jnp.eye(3)
constraints = [inside, 150.0 - spacing]

before = np.asarray(noise_at(houses, turbines).evaluate())
result = prp.optimize(objective, wrt=[pos], constraints=constraints, penalty_weight=10.0, n_steps=600, learning_rate=1.5)
final = np.asarray(result.params["pos"])
after = np.asarray(noise_at(houses, turbines).evaluate({"pos": final}))

print("noise at the houses before [dB]", np.round(before, 1))
print("noise at the houses after  [dB]", np.round(after, 1))
print("final positions [m]\n", np.round(final, 1))
print("inside margin [m] ", np.round(-np.asarray(area.signed_distance(prp.PointList(jnp.array(final), site)).evaluate()), 1))


def draw(th):
    # Figure, a noise map of the optimised layout on a 15 by 15 grid.
    grid = prp.Grid(site, (0.0, 850.0), (0.0, 750.0), nx=15, ny=15)
    fixed = prp.PointList(jnp.array(final), site)
    level_map = np.asarray(noise_at(grid.positions(), fixed).evaluate()).reshape(grid.shape())
    xs, ys = np.linspace(0.0, 850.0, 15), np.linspace(0.0, 750.0, 15)

    fig, (ax, bx) = st.figure(1, 2, height=3.9, gridspec_kw={"width_ratios": [1.35, 1]})
    cf = ax.contourf(xs, ys, level_map, levels=14, cmap=th.seq2)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02, label="noise [dB]")
    ring = np.vstack([area_xy, area_xy[:1]])
    ax.plot(ring[:, 0], ring[:, 1], color=th.brand2, lw=1.6, label="permitted area")
    ax.scatter(*start.T, marker="^", s=70, facecolor="none", edgecolor=th.fg, label="start")
    ax.scatter(*final.T, marker="^", s=110, color=th.brand, label="optimised", zorder=3)
    for a, b in zip(start, final):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color=th.fg, lw=0.8))
    ax.scatter(*houses_xy.T, marker="s", s=45, color=th.series[0], label="houses", zorder=3)
    st.map_axes(ax, th)
    ax.set(title="Optimised layout", xlabel="x [m]", ylabel="y [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=7.5)

    bx.plot(np.asarray(result.loss_history), color=th.brand)
    bx.set(xlabel="optimiser step", ylabel="loudest house plus penalty [dB]", title="Objective")
    return fig


st.save("07_design_optimization", draw)
