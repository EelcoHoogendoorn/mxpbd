"""a simply supported bridge of girders joined by splices

Each girder is a modally reduced truss;
each splice is a pair of point constraints at the top and bottom chord,
transmitting bending moment through the pair without any joint abstraction.
Released from its undeformed configuration under gravity,
the bridge sags and rings down on its global modes,
which emerge from a handful of local modes per girder.

The ring-down rate is the prescribed per-mode damping ratio, the same quantity
measured on real structures. Folded implicitly into the modal constraints
(see solve.modal_terms), any damping level is stable at any timestep,
and leaves the settled sag exactly untouched.
"""
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from modal_xpbd.body import ModalBody
from modal_xpbd.constraint import pin, pin_world
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.plot import draw_bodies, save_gif
from modal_xpbd.solve import step
from modal_xpbd.truss import girder

jax.config.update('jax_enable_x64', True)

n_girders = 8
n_cells = 4
gravity = (0.0, -1.0)
dt = 0.05
length = float(n_cells)
n_frames, steps_per_frame = 180, 2
damping = 64.0
stiffness = 6e5

shape = reduce_modes(girder(n_cells, stiffness=stiffness), n_modes=8)
bodies = [
	ModalBody.rest(shape, position=((i + 0.5) * length, 0.5), damping=damping)
	for i in range(n_girders)
] + [ModalBody.world()]
# splice pairs at each girder interface; supports at the outer bottom corners
constraints = [
	c
	for i in range(n_girders - 1)
	for c in (
		pin(bodies, i, i + 1, world_point=((i + 1) * length, 0.0)),
		pin(bodies, i, i + 1, world_point=((i + 1) * length, 1.0)),
	)
] + [
	pin_world(bodies, 0, world_point=(0.0, 0.0)),
	pin_world(bodies, n_girders - 1, world_point=(n_girders * length, 0.0)),
]

step_jit = jax.jit(step, static_argnames=('substeps',))

frames, tip = [], []
state = bodies
for f in range(n_frames):
	for i in range(steps_per_frame):
		state = step_jit(
			state, [constraints], dt=dt, substeps=1,
			gravity=gravity)
	frames.append(state)
	tip.append(state[n_girders // 2 - 1].world_points()[n_cells, 1])	# midspan bottom node
tip = np.asarray(tip)

fig, (ax, ax_t) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])
for i, f in enumerate(frames[:: n_frames // 8]):
	draw_bodies(ax, f, color='b', alpha=0.15 + 0.8 * i / 8)
draw_bodies(ax, bodies, color='r', alpha=0.4)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title(f'{n_girders} girders, {shape.n_modes} modes each, splice pairs; red: undeformed')

t = np.arange(n_frames) * dt * steps_per_frame
ax_t.plot(t, tip, c='k')
ax_t.set_xlabel('time')
ax_t.set_ylabel('midspan height')
fig.savefig(Path(__file__).parent / 'girder_bridge.png', dpi=120)

# animated gif
fig_anim, ax_anim = plt.subplots(figsize=(8, 2.2))


def draw_frame(i):
	ax_anim.clear()
	draw_bodies(ax_anim, bodies, color='r', alpha=0.25)
	draw_bodies(ax_anim, frames[i], color='b')
	ax_anim.set_xlim(-0.5, n_girders * length + 0.5)
	ax_anim.set_ylim(-1.6, 1.4)
	ax_anim.set_aspect('equal')
	ax_anim.set_xticks([])
	ax_anim.set_yticks([])


save_gif(fig_anim, draw_frame, n_frames, Path(__file__).parent / 'girder_bridge.gif', fps=20)

plt.show()
