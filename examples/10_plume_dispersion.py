"""A Gaussian plume under an uncertain wind.

A 30 m stack releases a gas continuously. The ground level concentration is
the ``steady`` Green's function of ``physics.superpose`` scaled by the
emission rate over the wind speed, with a reflecting ground and a mixing lid.
The wind speed (Weibull) and direction (von Mises) are random variables, so
the concentration at a school becomes a distribution, and the whole map
turns into a probability of exceeding a limit.
"""

import jax
import jax.numpy as jnp
import numpy as np

import prophys as prp
from prophys import physics
import _style as st

EMISSION = 40.0  # g/s
LIMIT = 150.0  # ug/m3
stack = np.array([[0.0, 0.0, 30.0]])
school = np.array([[900.0, 250.0, 1.5]])
spread = dict(spread_across=(0.22, 0.9), spread_vertical=(0.2, 0.85), vertical="reflect", lid=800.0, images=3)

speed_dist = prp.Weibull(scale=5.0, concentration=2.0)  # m/s at stack height
direction_dist = prp.VonMises(loc=np.deg2rad(15.0), kappa=5.0)  # direction the wind blows towards
speed = prp.RandomVariable("speed", speed_dist)
direction = prp.RandomVariable("direction", direction_dist)


def concentration(receptors, u, theta):
    """Ground level concentration [ug/m3] at the receptors."""
    green = physics.superpose(receptors, stack, [1.0], kind="steady", direction=theta, **spread)
    return 1e6 * EMISSION / prp.maximum(u, 0.5) * green


at_school = concentration(school, speed, direction)[..., 0]
exposure = prp.UncertainAttribute("school", prp.LogNormal(mu=prp.log(at_school + 1e-3), sigma=0.3), unit="ug/m3")
compiled = prp.ProbabilityModel(exposure).compile(n_samples=20_000, seed=3)

print(compiled.assess("school"))
print(compiled.assess("school", "quantile", level=0.99))
print(f"P(concentration at the school > {LIMIT:.0f} ug/m3) = {1 - float(compiled.prob('school', LIMIT)):.2%}")

# Maps on a 15 by 15 receptor grid, 225 receptors and one stack.
xs, ys = np.linspace(-300.0, 2200.0, 15), np.linspace(-1000.0, 1300.0, 15)
X, Y = np.meshgrid(xs, ys)
grid = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 1.5)])

# A closer 15 by 16 window along the mean wind for the 3D view, 240 receptors.
PX, PY = np.meshgrid(np.linspace(100.0, 2100.0, 15), np.linspace(-250.0, 750.0, 16))
near = np.column_stack([PX.ravel(), PY.ravel(), np.full(PX.size, 1.5)])
typical = np.asarray(concentration(near, 5.0, np.deg2rad(15.0)).evaluate()).reshape(PX.shape)

k1, k2 = jax.random.split(jax.random.PRNGKey(0))
u = np.asarray(speed_dist.sample(k1, (400,)).evaluate())
theta = np.asarray(direction_dist.sample(k2, (400,)).evaluate())
draws = np.asarray(concentration(grid, jnp.asarray(u)[:, None], jnp.asarray(theta)).evaluate())  # [400, 225]
p_exceed = (draws > LIMIT).mean(axis=0).reshape(X.shape)


def draw(th):
    fig = st.blank(8.2, 3.8)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0], projection="3d")
    ax.plot_surface(
        PX / 1000, PY / 1000, typical, cmap=th.seq, rstride=1, cstride=1, linewidth=0.25, edgecolor=th.surface, alpha=0.97
    )
    ax.set_xlabel("x [km]", fontsize=8)
    ax.set_ylabel("y [km]", fontsize=8)
    ax.view_init(elev=32, azim=-58)
    st.pane_3d(ax, th)
    ax.set_title("Concentration at 5 m/s wind [ug/m3]", loc="left")

    bx = fig.add_subplot(gs[1])
    cf = bx.contourf(X / 1000, Y / 1000, p_exceed, levels=np.linspace(0, p_exceed.max(), 12), cmap=th.seq2)
    fig.colorbar(cf, ax=bx, fraction=0.046, pad=0.02, format="%.2f", label=f"P(c > {LIMIT:.0f} ug/m3)")
    bx.scatter([0.0], [0.0], marker="^", s=70, color=th.fg, label="stack")
    bx.scatter([school[0, 0] / 1000], [school[0, 1] / 1000], marker="s", s=50, color=th.brand, label="school")
    st.map_axes(bx, th)
    bx.set(xlabel="x [km]", ylabel="y [km]", title="Exceedance probability, 400 winds")
    bx.legend(loc="lower right")
    return fig


st.save("10_plume_dispersion", draw)
