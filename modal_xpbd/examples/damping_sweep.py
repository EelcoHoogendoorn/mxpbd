"""the same girder, bent and released, across the damping ratio spectrum

Damping here is the per-mode damping ratio zeta,
the quantity measured on real structures. Each dashpot 2 zeta omega
is folded implicitly into its modal constraint (see solve.modal_terms),
and this example puts the consequences on display:

- every damping level runs at one and the same timestep. Explicit integration
  of a dashpot demands dt ~ 1 / (zeta omega): the heavier the damping, the
  smaller the step, and in direct FEM strong damping contributes numerical
  stiffness besides. The implicit fold has neither problem.
- each trace tracks the analytic damped oscillator release (dotted),
  from the undamped reference through overdamped creep; the residual
  amplitude and phase bias is the (omega h)**2 implicit euler signature,
  shrinking with substeps.
- the overdamped girder returns slower than the well-damped one:
  the classic overdamped creep.
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

zetas = {0.0: 'dimgray', 0.1: 'tab:blue', 0.7: 'tab:green', 4.0: 'tab:orange'}
mode = 0
amplitude = 2.0
n_frames, steps_per_frame, substeps = 100, 2, 64

shape = reduce_modes(girder(16, stiffness=400.0), n_modes=6)
omega = float(shape.omega[mode])
period = 2 * np.pi / omega
dt = period / 80
# bend the girder into its lowest mode, to be released from rest
bent = ModalBody.rest(shape).replace(
	amplitudes=jnp.zeros(shape.n_modes).at[mode].set(amplitude))

step_jit = jax.jit(step, static_argnames=('substeps',))


def simulate(zeta):
	bodies = [bent]
	frames, trace = [], []
	for f in range(n_frames):
		for s in range(steps_per_frame):
			bodies = step_jit(bodies, [], dt=dt, substeps=substeps, damping=zeta)
		frames.append(bodies)
		trace.append(float(bodies[0].amplitudes[mode]))
	return frames, np.asarray(trace)


def analytic(zeta, t):
	"""damped oscillator released from rest at `amplitude`"""
	d = zeta * omega
	if zeta < 1:
		wd = omega * np.sqrt(1 - zeta ** 2)
		return amplitude * np.exp(-d * t) * (np.cos(wd * t) + d * np.sin(wd * t) / wd)
	s = omega * np.sqrt(zeta ** 2 - 1)
	return amplitude * np.exp(-d * t) * (np.cosh(s * t) + d * np.sinh(s * t) / s)


runs = {zeta: simulate(zeta) for zeta in zetas}

t = np.arange(1, n_frames + 1) * dt * steps_per_frame
fig, ax_t = plt.subplots(figsize=(10, 4))
for zeta, color in zetas.items():
	ax_t.plot(t / period, runs[zeta][1], color=color, label=f'zeta = {zeta}')
	ax_t.plot(t / period, analytic(zeta, t), color='k', ls=':', lw=1.0)
ax_t.set_xlabel('time / period')
ax_t.set_ylabel(f'amplitude of mode {mode}')
ax_t.set_title('girder released from a bend, every damping ratio at the same timestep; dotted: analytic oscillator')
ax_t.legend()
fig.savefig(Path(__file__).parent / 'damping_sweep.png', dpi=120)

# animated gif: the ring-downs stacked
fig_anim, ax_anim = plt.subplots(figsize=(7, 5.6))
spacing = 2.2


def draw_frame(i):
	ax_anim.clear()
	for row, (zeta, color) in enumerate(zetas.items()):
		offset = (0.0, -spacing * row)
		draw_bodies(ax_anim, [ModalBody.rest(shape, position=offset)], color='r', alpha=0.2)
		draw_bodies(ax_anim, [runs[zeta][0][i][0].replace(position=jnp.asarray(offset))], color=color)
		ax_anim.text(3.3, -spacing * row, f'zeta = {zeta}', color=color, va='center')
	ax_anim.set_xlim(-8, 8)
	ax_anim.set_ylim(-spacing * (len(zetas) - 1) - 1.3, 1.3)
	ax_anim.set_aspect('equal')
	ax_anim.set_xticks([])
	ax_anim.set_yticks([])


save_gif(fig_anim, draw_frame, n_frames, Path(__file__).parent / 'damping_sweep.gif', fps=25)

plt.show()
