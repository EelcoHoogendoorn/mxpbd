"""free vibration of a single floating modal body

Each mode is both a degree of freedom and an elastic constraint:
the xpbd projection onto the modal constraint, plus the position-velocity update,
is the oscillator. Kicked with initial modal rates and left alone,
the body rings at its eigenfrequencies, while its frame stays put:
mass-orthonormal modes carry no linear or angular momentum.

Note the substeps: the single-iteration projection is implicit-euler flavored
and dissipates (omega dt)**2 log energy per step; substepping shrinks it away.
"""
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from modal_xpbd.body import ModalBody
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.plot import draw_bodies, save_gif
from modal_xpbd.solve import step
from modal_xpbd.truss import girder

jax.config.update('jax_enable_x64', True)

shape = reduce_modes(girder(6, stiffness=400.0), n_modes=6)
omega = np.asarray(shape.omega)
print('eigenfrequencies:', omega.round(2))

# kick the two lowest modes; rate = amplitude * omega
kicked = {0: 0.8, 1: 0.4}
rates = jnp.zeros(shape.n_modes)
for mode, amplitude in kicked.items():
	rates = rates.at[mode].set(amplitude * omega[mode])
bodies = [ModalBody.rest(shape).replace(rates=rates)]

dt = 2 * np.pi / omega[0] / 60		# 60 frames per fundamental period
n_frames = 240
step_jit = jax.jit(step, static_argnames=('substeps',))

frames, amplitudes = [], []
state = bodies
for i in range(n_frames):
	state = step_jit(state, [], dt=dt, substeps=8)
	frames.append(state)
	amplitudes.append(np.asarray(state[0].amplitudes))
amplitudes = np.asarray(amplitudes)

fig, (ax, ax_t) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[2, 1])
for i, f in enumerate(frames[: 60 : 10]):
	draw_bodies(ax, f, color='b', alpha=0.15 + 0.8 * i / 6)
draw_bodies(ax, bodies, color='r', alpha=0.4)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('free floating body ringing on its kicked modes; red: undeformed')

t = np.arange(n_frames) * dt
for mode in kicked:
	ax_t.plot(t, amplitudes[:, mode], label=f'mode {mode}, T={2 * np.pi / omega[mode]:.2f}')
ax_t.set_xlabel('time')
ax_t.set_ylabel('modal amplitude')
ax_t.legend()
fig.savefig(Path(__file__).parent / 'free_vibration.png', dpi=120)

fig_anim, ax_anim = plt.subplots(figsize=(6, 2.4))


def draw_frame(i):
	ax_anim.clear()
	draw_bodies(ax_anim, bodies, color='r', alpha=0.25)
	draw_bodies(ax_anim, frames[i], color='b')
	ax_anim.set_xlim(-4.0, 4.0)
	ax_anim.set_ylim(-1.6, 1.6)
	ax_anim.set_aspect('equal')
	ax_anim.set_xticks([])
	ax_anim.set_yticks([])


save_gif(fig_anim, draw_frame, n_frames, Path(__file__).parent / 'free_vibration.gif', fps=25)

plt.show()
