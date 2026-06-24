"""buckling of a spliced girder beam under a prescribed end displacement

A horizontal beam of girders, clamped to the world at both ends by pin pairs;
the right pair of anchors is driven inwards, crushing the beam like a test rig.

There is no imperfection or bias term of any kind: the symmetry is broken by
floating point rounding alone, which makes for the cleanest possible break.
The pre-buckle compression is imperceptible, until the prescribed displacement
crosses critical and the beam pops. Note that the observed pop trails the euler
line somewhat: the straight state remains a true (unstable) equilibrium past
critical, and the dwell is the e-folding time for rounding level noise to grow
visible. A bias term would make takeoff hug the euler line instead,
at the cost of rounding off the break.

A single linearized body cannot buckle: geometric stiffness requires the
coupling of axial load into lateral deflection, which a constant stiffness
matrix has linearized away. The chain of modal bodies recovers it at the
assembly level, since the large rotations live in the floating frames.

Note that this demo does not use substepping, nor implicit integration, nor a global solver;
merely quasi-explicit integration with large timesteps, and local constraint solves.
Yet it can simulate arbitrarily stiff materials and geometries (this would be an accurate model of a metal wire).

In the limit to zero compliance, one gracefully retrieves the properties of XPBD; stable rigid bodies;
yet within the same simulation, one can support arbitrarily compliant bodies,
and arbitrary deformational complexity, should one be willing to allocate DOFs to that.
"""
import dataclasses
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from modal_xpbd.body import ModalBody
from modal_xpbd.constraint import pin, pin_world
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.plot import draw_bodies, save_gif
from modal_xpbd.solve import step
from modal_xpbd.truss import girder

jax.config.update('jax_enable_x64', True)

n_girders = 8
n_cells = 32
stiffness = 1e11
length = float(n_cells)
span = n_girders * length
dt = 0.04
substeps = 20
n_frames, steps_per_frame = 120, 1
crush = 0.15		# prescribed end displacement; quartic in time, so the pop lands early-mid
gravity = (0.0, 0.0)

shape = reduce_modes(girder(n_cells, stiffness=stiffness), n_modes=8)
bodies = [
	ModalBody.rest(shape, position=((i + 0.5) * length, 0.0), damping=3)
	for i in range(n_girders)
] + [ModalBody.world()]
splices = [
	c
	for i in range(n_girders - 1)
	for c in (
		pin(bodies, i, i + 1, world_point=((i + 1) * length, -0.5)),
		pin(bodies, i, i + 1, world_point=((i + 1) * length, +0.5)),
	)
] + [
	pin_world(bodies, 0, world_point=(0.0, -0.5)),
	pin_world(bodies, 0, world_point=(0.0, +0.5)),
]
crosshead = [
	pin_world(bodies, n_girders - 1, world_point=(span, -0.5)),
	pin_world(bodies, n_girders - 1, world_point=(span, +0.5)),
]


def crushed(d):
	"""the crosshead anchors, driven inwards by d; anchors are data, no retrace"""
	return [dataclasses.replace(c, anchor_b=c.anchor_b - jnp.asarray([d, 0.0])) for c in crosshead]


EI = stiffness / 2		# two chords at +-1/2 from the neutral axis
axial = 2 * stiffness / span		# chord contribution to beam axial stiffness
critical = 4 * np.pi ** 2 * EI / span ** 2 / axial	# clamped-clamped, as displacement
print(f'estimated critical displacement: {critical:.4f} of a crush of {crush}')

step_jit = jax.jit(step, static_argnames=('substeps',))

frames, bow, applied = [], [], []
state = bodies
for f in range(n_frames):
	d = crush * (f / n_frames)
	constraints = splices + crushed(d)
	for i in range(steps_per_frame):
		state = step_jit(
			state, [constraints], dt=dt, substeps=substeps,
			gravity=gravity)
	frames.append(state)
	bow.append(float((state[n_girders // 2 - 1].position[1] + state[n_girders // 2].position[1]) / 2))
	applied.append(d)
bow, applied = np.asarray(bow), np.asarray(applied)

fig, (ax, ax_t) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[1, 1.4])
for i, f in enumerate(frames[:: n_frames // 10]):
	draw_bodies(ax, f, alpha=0.12 + 0.85 * i / 10)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('prescribed end displacement; pop past critical, seeded by rounding noise')

ax_t.plot(applied, np.abs(bow), c='k')
ax_t.axvline(critical, ls=':', c='crimson', label='euler estimate')
ax_t.set_xlabel('prescribed end displacement')
ax_t.set_ylabel('midspan deflection')
ax_t.legend()
fig.savefig(Path(__file__).parent / 'buckling.png', dpi=120)

# animated gif
fig_anim, ax_anim = plt.subplots(figsize=(10, 3))


def draw_frame(i):
	ax_anim.clear()
	draw_bodies(ax_anim, frames[i])
	ax_anim.set_xlim(-4.0, span + 4.0)
	ax_anim.set_ylim(-4.0, 4.0)
	ax_anim.set_aspect('equal')
	ax_anim.set_xticks([])
	ax_anim.set_yticks([])


save_gif(fig_anim, draw_frame, n_frames, Path(__file__).parent / 'buckling.gif', fps=50)

plt.show()
