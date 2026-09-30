"""A raster field and a route through it, rendered in 3D.

A 14 by 14 elevation raster becomes a ``Field`` with differentiable bilinear
interpolation. A road is a ``Polyline`` between two fixed ends with two free
middle vertices, the initial guess passing over the main hill. The elevation is read at 40 points along the road, and the
middle vertices move by gradient descent to trade the climb against the
length of the road.
"""

import jax.numpy as jnp
import numpy as np
from matplotlib.colors import LightSource

import prophys as prp
import _style as st

site = prp.Frame("site", units="m")
CELL = 50.0
n = 14
xs = np.arange(n) * CELL
X, Y = np.meshgrid(xs, xs)


def hill(x0, y0, h, r):
    return h * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * r**2))


elevation = 40.0 + hill(330.0, 330.0, 90.0, 120.0) + hill(520.0, 150.0, 45.0, 90.0) + hill(130.0, 520.0, 55.0, 110.0) + 0.04 * X
terrain = prp.Field(jnp.array(elevation), site, origin=(0.0, 0.0), cell=CELL)  # 196 cells

start, end = jnp.array([[20.0, 30.0]]), jnp.array([[630.0, 620.0]])
middle = prp.Param("middle", shape=(2, 2), init=jnp.array([[230.0, 200.0], [450.0, 410.0]]))
road = prp.Polyline(prp.symbolic.concatenate([start, middle, end], axis=0), site)  # 4 vertices

t = jnp.linspace(0.0, 1.0, 40)
profile = terrain.interp(prp.PointList(road.point_at(t), site))  # 40 points, 240 objects in total
climb = prp.sum_(prp.relu(profile[1:] - profile[:-1]))  # metres of ascent
objective = climb + 0.05 * road.length()

before = {"climb": float(climb.evaluate()), "length": float(road.length().evaluate())}
result = prp.optimize(objective, wrt=[middle], n_steps=400, learning_rate=4.0)
best = {"middle": result.params["middle"]}
after = {"climb": float(climb.evaluate(best)), "length": float(road.length().evaluate(best))}
print(f"initial road    climb {before['climb']:6.1f} m   length {before['length']:6.1f} m")
print(f"optimised road  climb {after['climb']:6.1f} m   length {after['length']:6.1f} m")

# 3D rendering of the terrain with both roads draped on it.

def draw(th):
    fig = st.blank(7.4, 4.4)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    light = LightSource(azdeg=315, altdeg=40)
    rgb = light.shade(elevation, cmap=th.brand_map, vert_exag=2.0, blend_mode="soft")
    ax.plot_surface(X, Y, elevation, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    for params, color, label in (({}, th.series[1], "initial road"), (best, "#ffffff", "optimised road")):
        xy = np.asarray(road.point_at(t).evaluate(params))
        z = np.asarray(profile.evaluate(params)) + 3.0
        ax.plot(xy[:, 0], xy[:, 1], z, color=color, lw=2.6, label=label, zorder=10)
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="elevation [m]")
    ax.view_init(elev=38, azim=-62)
    ax.set_box_aspect((1, 1, 0.38))
    st.pane_3d(ax, th)
    ax.set_title("Road over a Field, before and after optimisation", loc="left")
    ax.legend(loc="upper right", frameon=True, facecolor=th.surface, edgecolor=th.grid, framealpha=0.9)
    return fig


st.save("09_terrain_route_3d", draw)
