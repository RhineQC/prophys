"""Symbolic expressions and gradients.

Every quantity in prophys is a node of a graph. Parameters are named leaves,
arithmetic builds the graph and ``evaluate`` computes it with JAX. Because
the result is a pure JAX function, ``jax.grad`` and ``jax.vmap`` work on it
directly.
"""

import jax
import jax.numpy as jnp

import prophys as prp
import _style as st

a = prp.Param("a", init=2.0)
b = prp.Const(3.0)
expr = a * b + prp.exp(-a) * prp.sin(3 * a)

print(f"expression at the initial value a = 2    {float(expr.evaluate()):.4f}")
print(f"expression at a = 0.5                    {float(expr.evaluate({'a': 0.5})):.4f}")

# A bounded parameter lives in (0, 1) whatever the optimiser does to it.
share = prp.Param("share", init=0.3, bounds=(0.0, 1.0))
print(f"bounded parameter at its initial value   {float(share.evaluate()):.4f}")

# Derivatives come from JAX.


def f(v):
    """The expression as a plain function of a."""
    return expr.evaluate({"a": v})


df = jax.grad(f)
print(f"d expr / d a at a = 0.5                  {float(df(0.5)):.4f}")

# The smooth minimum used in optimisation mode against the exact one.
d = prp.stack([prp.Param("d1", init=4.0), prp.Const(5.0)])
print(f"exact minimum                           {float(prp.minimum(d[0], d[1]).evaluate()):.4f}")
print(f"softmin with tau = 0.5                   {float(prp.softmin(d, tau=0.5).evaluate()):.4f}")


def draw(th):
    # Figure, the expression and its derivative over a range of a.
    grid = jnp.linspace(0.0, 3.0, 300)
    values = jax.vmap(f)(grid)
    slopes = jax.vmap(df)(grid)
    fig, (ax, bx) = st.figure(1, 2)
    ax.plot(grid, values, color=th.brand)
    ax.set(xlabel="a", ylabel="value", title="3a + exp(-a) sin(3a)")
    bx.axhline(0.0, color=th.muted, lw=0.8)
    bx.plot(grid, slopes, color=th.brand2)
    bx.set(xlabel="a", ylabel="derivative", title="Its exact gradient from jax.grad")
    return fig


st.save("02_symbolic_expressions", draw)
