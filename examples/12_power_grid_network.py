"""A small power grid under uncertain wind, with reinforcement decisions.

Twelve buses form a meshed grid. A wind farm in one corner feeds uncertain
power, the loads are uncertain too, and a DC power flow gives the line flows
through ``physics.network_potentials``. Each line has a ``BinaryState`` that
decides whether it is reinforced, which doubles its susceptance and its
rating. Gradient descent on the relaxed states trades expected overload
against the cost of reinforcement, and the final hard decisions are read off
the same graph.
"""

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import prophys as prp
from prophys import physics
import _style as st

site = prp.Frame("grid", units="km")
nx, ny = 4, 3
xy = np.array([[20.0 * i, 20.0 * j] for j in range(ny) for i in range(nx)])
edges = np.array(
    [[j * nx + i, j * nx + i + 1] for j in range(ny) for i in range(nx - 1)]
    + [[j * nx + i, (j + 1) * nx + i] for j in range(ny - 1) for i in range(nx)]
    + [[1, 6], [6, 11]]
)
grid = prp.Network(prp.PointList(jnp.array(xy), site), jnp.array(edges), site)  # 12 buses
length = np.asarray(grid.edge_lengths().evaluate())
base_b = 400.0 / length  # susceptance, per unit
rating = np.full(len(edges), 1.3)  # per unit

# Monte Carlo scenarios of the injections, wind at bus 11 and loads elsewhere.
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
n_s = 400
wind = 6.0 * np.clip(np.asarray(prp.Weibull(scale=0.55, concentration=2.2).sample(k1, (n_s,)).evaluate()), 0.0, 1.0)
loads = np.asarray(prp.Gaussian(0.55, 0.12).sample(k2, (n_s, 11)).evaluate())
injection = np.zeros((n_s, 12))
injection[:, 1:] = -loads
injection[:, 11] += wind
injection[:, 0] = -injection[:, 1:].sum(axis=1)  # bus 0 is the slack

states = [prp.BinaryState(f"reinforce_{k}", init=0.2, mode="opt", tau=0.5) for k in range(len(edges))]
upgrade = prp.stack(states)
b = (1.0 + upgrade) * base_b  # expression first, so NumPy does not broadcast over it
theta = physics.network_potentials(jnp.array(edges), b, jnp.array(injection), ground=0)  # [n_s, 12]
# Incidence matrix, flow on each line from the angle difference of its buses.
incidence = np.zeros((12, len(edges)))
incidence[edges[:, 0], np.arange(len(edges))] = 1.0
incidence[edges[:, 1], np.arange(len(edges))] = -1.0
flow = b * (theta @ jnp.array(incidence))
loading = prp.abs_(flow) / ((1.0 + upgrade) * rating)
overload = prp.mean(prp.sum_(prp.softplus(20.0 * (loading - 1.0)) / 20.0, axis=1))
cost = 0.002 * prp.sum_(upgrade * length)
objective = overload + cost


def report(env):
    ld = np.asarray(loading.evaluate(env))
    return (ld > 1.0).mean(axis=0)


as_built = {state.logit.name: -1e3 for state in states}  # every state hard off
p_before = report(as_built)
result = prp.optimize(objective, wrt=[s.logit for s in states], n_steps=300, learning_rate=0.1)
chosen = dict(result.params)
hard = [int(float(v) > 0.0) for v in chosen.values()]
decided = {name: (1e3 if h else -1e3) for name, h in zip(chosen, hard)}
p_after = report(decided)
print("reinforced lines", [tuple(e) for e, h in zip(edges.tolist(), hard) if h])
print(f"worst line overload probability before {p_before.max():.1%}, after {p_after.max():.1%}")
print(f"expected overload before {float(overload.evaluate(as_built)):.4f}, after {float(overload.evaluate(decided)):.4f}")


def draw(th):
    fig, axes = st.figure(1, 2, height=3.2)
    cmap = th.seq2
    for ax, p, title in ((axes[0], p_before, "Overload probability, as built"), (axes[1], p_after, "Thick lines reinforced")):
        for k, (i, j) in enumerate(edges):
            lw = 2.0 + 4.0 * (hard[k] if ax is axes[1] else 0)
            ax.plot(*xy[[i, j]].T, color=cmap(0.15 + 0.85 * min(p[k] / 0.3, 1.0)), lw=lw, solid_capstyle="round", zorder=1)
        ax.scatter(*xy.T, s=40, color=th.fg, zorder=2)
        ax.scatter(*xy[[11]].T, s=160, marker="*", color=th.brand2, zorder=3, label="wind farm")
        ax.scatter(*xy[[0]].T, s=70, marker="s", color=th.brand, zorder=3, label="slack bus")
        st.map_axes(ax, th)
        ax.set(title=title, xlabel="x [km]", ylabel="y [km]", xlim=(-6, 66), ylim=(-6, 46))
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5, ncol=2)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 0.3))
    fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02, shrink=0.8, label="P(line overloaded)")
    return fig


st.save("12_power_grid_network", draw)
