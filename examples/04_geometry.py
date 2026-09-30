"""Geometry in a frame.

Points, polygons and grids live in a ``Frame``, a shared coordinate system.
Every geometric query returns an expression, so distances, areas and
containment feed the rest of a model and stay differentiable. Hard queries
are exact, and their smooth counterparts are what an optimiser needs.
"""

import jax.numpy as jnp
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import prophys as prp
import _style as st

site = prp.Frame("site", units="m")
zone = prp.Polygon(
    jnp.array([[120.0, 80.0], [300.0, 60.0], [380.0, 200.0], [300.0, 330.0], [150.0, 300.0], [80.0, 190.0]]),
    site,
)
wells = prp.PointList(jnp.array([[60.0, 380.0], [420.0, 360.0], [440.0, 60.0], [230.0, 190.0], [40.0, 40.0]]), site)

print(f"zone area {float(zone.area().evaluate()):.0f} m2, perimeter {float(zone.perimeter().evaluate()):.0f} m")
print("wells inside the zone ", np.asarray(zone.contains(wells).evaluate()).astype(int))
print("signed distance [m]   ", np.round(np.asarray(zone.signed_distance(wells).evaluate()), 1))

# A 14 by 14 evaluation grid, 196 points, well inside the free tier.
grid = prp.Grid(site, (0.0, 460.0), (0.0, 420.0), nx=14, ny=14)
points = grid.positions()
signed = np.asarray(zone.signed_distance(points).evaluate()).reshape(grid.shape())
to_well = np.asarray(points.closest_distance(wells).evaluate()).reshape(grid.shape())
xs, ys = np.linspace(0.0, 460.0, 14), np.linspace(0.0, 420.0, 14)

# Hard and smooth containment along a line through the zone.
transect = prp.PointList(jnp.stack([jnp.linspace(0.0, 460.0, 120), jnp.full(120, 190.0)], axis=1), site)
hard = np.asarray(zone.contains(transect, tau=0.0).evaluate())
soft = {tau: np.asarray(zone.contains(transect, tau=tau).evaluate()) for tau in (5.0, 20.0)}


def draw(th):
    fig, (ax, bx) = st.figure(1, 2, height=3.6)
    cf = ax.contourf(xs, ys, signed, levels=np.arange(-140.0, 221.0, 20.0), cmap=th.div, norm=TwoSlopeNorm(0.0, -140.0, 220.0))
    ax.contour(xs, ys, to_well, levels=[60.0], colors=th.brand2, linewidths=1.2, linestyles="--")
    v = np.asarray(zone.evaluate())
    ax.fill(v[:, 0], v[:, 1], fill=False, ec=th.fg, lw=1.6)
    w = np.asarray(wells.evaluate())
    ax.scatter(w[:, 0], w[:, 1], s=40, color=th.brand, zorder=3, label="wells")
    ax.axhline(190.0, color=th.muted, lw=0.8, ls=":")
    st.map_axes(ax, th)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02, label="signed distance to zone [m]")
    ax.set(title="Signed distance to the zone", xlabel="x [m]", ylabel="y [m]")
    ax.legend(loc="lower center", framealpha=0.8, frameon=True)

    x = np.linspace(0.0, 460.0, 120)
    bx.plot(x, hard, color=th.fg, lw=1.6, label="exact, eval mode")
    bx.plot(x, soft[5.0], color=th.series[0], label="smooth, tau 5 m")
    bx.plot(x, soft[20.0], color=th.series[1], label="smooth, tau 20 m")
    bx.set(xlabel="x along the dotted line [m]", ylabel="inside the zone", title="Hard and smooth containment")
    bx.legend(loc="center")
    return fig


st.save("04_geometry", draw)
