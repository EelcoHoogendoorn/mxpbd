"""the girder bridge at three levels of girder stiffness

The whole setup is built and run three times.
The rigid run carries the same modes with their compliance zeroed,
which is exactly a rigid body: a chain of rigid girders with
moment carrying splices does not deflect at all, up to solver imperfection.
All flexibility on display is modal: at small amplitude, deflection scales
linearly with compliance and period with its square root,
with assembly level geometric effects taking over at large amplitude.
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
damping = 0.1
stiffness = 6e5

step_jit = jax.jit(step, static_argnames=('substeps',))


def simulate(shape):
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

	frames, tip = [], []
	state = bodies
	for i in range(n_frames):
		for j in range(steps_per_frame):
			state = step_jit(
				state, [constraints], dt=dt, substeps=2,
				gravity=gravity)
		frames.append(state)
		tip.append(state[n_girders // 2 - 1].world_points()[n_cells, 1])	# midspan bottom node
	return frames, np.asarray(tip)


base = reduce_modes(girder(n_cells, stiffness=stiffness), n_modes=8)
soft = reduce_modes(girder(n_cells, stiffness=stiffness / 4), n_modes=8)
setups = {
	'rigid': (base.rigid(), 'dimgray'),
	'EA': (base, 'tab:blue'),
	'EA / 4': (soft, 'tab:green'),
}
runs = {name: simulate(shape) for name, (shape, color) in setups.items()}

fig, (ax, ax_t) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])
for name, (_, color) in setups.items():
	draw_bodies(ax, runs[name][0][-1], color=color)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title(f'{n_girders} girders, 8 modes each, splice pairs; settled, per girder stiffness')

t = np.arange(n_frames) * dt * steps_per_frame
for name, (_, color) in setups.items():
	ax_t.plot(t, runs[name][1], color=color, label=name)
ax_t.set_xlabel('time')
ax_t.set_ylabel('midspan height')
ax_t.legend()
fig.savefig(Path(__file__).parent / 'stiffness_sweep.png', dpi=120)

# animated gif
fig_anim, ax_anim = plt.subplots(figsize=(8, 2.2))


def draw_frame(i):
	ax_anim.clear()
	for name, (_, color) in setups.items():
		draw_bodies(ax_anim, runs[name][0][i], color=color, alpha=0.4 if name == 'rigid' else 1.0)
	ax_anim.set_xlim(-0.5, n_girders * length + 0.5)
	ax_anim.set_ylim(-1.6, 1.6)
	ax_anim.set_aspect('equal')
	ax_anim.set_xticks([])
	ax_anim.set_yticks([])


save_gif(fig_anim, draw_frame, n_frames, Path(__file__).parent / 'stiffness_sweep.gif', fps=20)

plt.show()
