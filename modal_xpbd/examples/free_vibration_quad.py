"""free vibration of a single solid (Q4 continuum) body — a validation probe

The truss free_vibration shows the modal oscillator; this one swaps the lattice
for a plane-stress continuum strip and checks the reduced eigenfrequencies
against the analytic free-free continuum values: Euler-Bernoulli bending,
(beta_n L)^2 sqrt(EI / rho A L^4) with beta_n L = 4.730, 7.853, 10.996, ...,
and longitudinal modes n pi c / L with c = sqrt(E / rho).

If a single free body lands on these, the continuum + modal reduction is sound,
and any stiffness surprise in an assembled, spliced structure must come from the
coupling, not the body. Kicked and left alone, it rings while its frame stays
put: mass-orthonormal modes carry no linear or angular momentum.
"""
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from modal_xpbd.body import ModalBody
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.plot import draw_bodies, save_gif
from modal_xpbd.quad import quad_strip
from modal_xpbd.solve import step

jax.config.update('jax_enable_x64', True)

E, nu, density = 400.0, 0.3, 1.0
length, height, cell = 6.0, 1.0, 0.5
shape = reduce_modes(
	quad_strip(nx=round(length / cell), ny=round(height / cell), cell=cell, E=E, nu=nu, density=density),
	n_modes=6)

# analytic free-free continuum frequencies for comparison
EI, rhoA, c = E * height ** 3 / 12, density * height, np.sqrt(E / density)
bending = np.array([4.730, 7.853, 10.996]) ** 2 * np.sqrt(EI / (rhoA * length ** 4))
axial = np.arange(1, 3) * np.pi * c / length
print('reduced eigenfrequencies:', np.asarray(shape.omega).round(2))
print('analytic bending       :', bending.round(2))
print('analytic longitudinal  :', axial.round(2))

omega = np.asarray(shape.omega)
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
	draw_bodies(ax, f, alpha=0.15 + 0.8 * i / 6)
draw_bodies(ax, bodies, color='r', alpha=0.4)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('free floating solid strip ringing on its kicked modes; red: undeformed')

t = np.arange(n_frames) * dt
for mode in kicked:
	ax_t.plot(t, amplitudes[:, mode], label=f'mode {mode}, T={2 * np.pi / omega[mode]:.2f}')
ax_t.set_xlabel('time')
ax_t.set_ylabel('modal amplitude')
ax_t.legend()
fig.savefig(Path(__file__).parent / 'free_vibration_quad.png', dpi=120)

fig_anim, ax_anim = plt.subplots(figsize=(6, 2.4))


def draw_frame(i):
	ax_anim.clear()
	draw_bodies(ax_anim, bodies, color='r', alpha=0.25)
	draw_bodies(ax_anim, frames[i])
	ax_anim.set_xlim(-4.0, 4.0)
	ax_anim.set_ylim(-1.6, 1.6)
	ax_anim.set_aspect('equal')
	ax_anim.set_xticks([])
	ax_anim.set_yticks([])


save_gif(fig_anim, draw_frame, n_frames, Path(__file__).parent / 'free_vibration_quad.gif', fps=25)

plt.show()
