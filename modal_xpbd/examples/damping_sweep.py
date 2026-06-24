"""a compound structure, bent and released, across the damping ratio spectrum

Where free_vibration shows a single modal oscillator damped, this shows the same
undamped -> critically damped -> overdamped behaviour reached by a *compound*
structure: a cantilever of three flexible girders, spliced into one beam, clamped
at the base, sagged a little under gravity and released.

The model's damping input is the per-body modal damping ratio. A body's own modes
are fast (its free-free frequencies); the assembled beam's bending fundamental is
slow. A local dashpot 2 zeta_local omega_local, seen by that slow global mode,
contributes an effective global damping

	zeta_global = zeta_local * Omega / omega_local

the local damping diluted by the frequency ratio (which grows like n_bodies**2 as
the chain lengthens). So to damp the *structure* at a target ratio, the per-body
damping is scaled up by that same ratio omega_local / Omega, measured here from
the undamped ring-down. Scaled so, the tip rings down on the analytic damped
oscillator (dotted), from the undamped reference through critical to overdamped
creep -- high, even critical, damping of a multi-body assembly, all at one and the
same timestep, where explicit integration of the dashpots would demand
dt ~ 1 / (zeta_local omega_local) and collapse.
"""
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

n_bodies = 3
length = 8.0
shape = reduce_modes(girder(8, stiffness=1e5), n_modes=6)
omega_local = float(shape.omega[0])			# a single body's fundamental
zetas = [0.0, 0.3, 1.0, 4.0]				# global damping targets
dt = 0.05
n_frames, steps_per_frame, substeps = 120, 4, 4
step_jit = jax.jit(step, static_argnames=('substeps',))


def build(damping):
	"""a cantilever chain: girders spliced into a beam, clamped to the world at the base"""
	bodies = [
		ModalBody.rest(shape, position=((i + 0.5) * length, 0.0), damping=damping)
		for i in range(n_bodies)
	] + [ModalBody.world()]
	splices = [
		c
		for i in range(n_bodies - 1)
		for c in (
			pin(bodies, i, i + 1, world_point=((i + 1) * length, -0.5)),
			pin(bodies, i, i + 1, world_point=((i + 1) * length, +0.5)),
		)
	] + [
		pin_world(bodies, 0, world_point=(0.0, -0.5)),
		pin_world(bodies, 0, world_point=(0.0, +0.5)),
	]
	return bodies, splices


# bend: sag the cantilever a little under gravity, to a deflected rest state.
# heavy damping settles it fast; the static sag itself is damping independent
bodies, splices = build(damping=50.0)
state = bodies
for _ in range(1000):
	state = step_jit(state, [splices], dt=dt, substeps=substeps, gravity=(0.0, -0.3))
amplitude = float(state[n_bodies - 1].position[1])		# tip deflection, the release amplitude
bent = [b.replace(velocity=b.velocity * 0, angular_velocity=b.angular_velocity * 0, rates=b.rates * 0)
		for b in state]


def simulate(zeta_local):
	# release the bent structure from rest, every body at modal damping zeta_local
	state = [b.replace(damping=jnp.asarray(zeta_local)) for b in bent]
	frames, trace = [], []
	for f in range(n_frames):
		for s in range(steps_per_frame):
			state = step_jit(state, [splices], dt=dt, substeps=substeps)
		frames.append(state)
		trace.append(float(state[n_bodies - 1].position[1]))
	return frames, np.asarray(trace)


# the global bending fundamental, from the undamped release; the per-body damping
# needed for a target global ratio is then that ratio times omega_local / Omega
undamped = simulate(0.0)
t = np.arange(1, n_frames + 1) * dt * steps_per_frame
sign = np.sign(undamped[1])
ups = np.nonzero((sign[1:] > 0) & (sign[:-1] <= 0))[0]
Omega = 2 * np.pi / (np.diff(ups).mean() * dt * steps_per_frame)
ratio = omega_local / Omega
print(f'omega_local = {omega_local:.2f},  global Omega = {Omega:.3f},  ratio = {ratio:.1f} (~ n_bodies**2 with a shape factor)')

runs = {zeta: (undamped if zeta == 0.0 else simulate(zeta * ratio)) for zeta in zetas}


def analytic(zeta, t):
	"""damped oscillator released from rest at `amplitude`, at global ratio zeta"""
	d = zeta * Omega
	if zeta < 1:
		wd = Omega * np.sqrt(1 - zeta ** 2)
		return amplitude * np.exp(-d * t) * (np.cos(wd * t) + d * np.sin(wd * t) / wd)
	if zeta == 1:
		return amplitude * np.exp(-Omega * t) * (1 + Omega * t)
	s = Omega * np.sqrt(zeta ** 2 - 1)
	return amplitude * np.exp(-d * t) * (np.cosh(s * t) + d * np.sinh(s * t) / s)


fig, ax_t = plt.subplots(figsize=(10, 4))
for zeta in zetas:
	ax_t.plot(t / (2 * np.pi / Omega), runs[zeta][1], label=f'zeta_global = {zeta}')
	ax_t.plot(t / (2 * np.pi / Omega), analytic(zeta, t), color='k', ls=':', lw=1.0)
ax_t.set_xlabel('time / global period')
ax_t.set_ylabel('tip deflection')
ax_t.set_title(f'{n_bodies}-girder cantilever, bent and released; every damping ratio at one timestep; dotted: analytic')
ax_t.legend()
fig.savefig(Path(__file__).parent / 'damping_sweep.png', dpi=120)

# animated gif: the ring-downs stacked, the structure flexing. the deflection is
# a few percent of the span, so it is exaggerated here for visibility only (the
# trace plot above is the quantitative exhibit)
fig_anim, ax_anim = plt.subplots(figsize=(6, 7))
span = n_bodies * length
viz = 5.0
spacing = 2.0 * viz * abs(amplitude)


def exaggerate(b, j, base_y):
	rest = jnp.asarray([(j + 0.5) * length, 0.0])
	return b.replace(
		position=jnp.asarray([rest[0], base_y]) + viz * (b.position - rest),
		angle=viz * b.angle,
		amplitudes=viz * b.amplitudes,
	)


def draw_frame(i):
	ax_anim.clear()
	for row, zeta in enumerate(zetas):
		base_y = -spacing * row
		rest = [ModalBody.rest(shape, position=((j + 0.5) * length, base_y)) for j in range(n_bodies)]
		draw_bodies(ax_anim, rest, color='r', alpha=0.15)
		draw_bodies(ax_anim, [exaggerate(b, j, base_y) for j, b in enumerate(runs[zeta][0][i][:-1])])
		ax_anim.text(span + 1.0, base_y, f'zeta = {zeta}', color='k', va='center')
	ax_anim.set_xlim(-1.0, span + 5.0)
	ax_anim.set_ylim(-spacing * (len(zetas) - 1) - spacing, spacing)
	ax_anim.set_aspect('equal')
	ax_anim.set_xticks([])
	ax_anim.set_yticks([])


save_gif(fig_anim, draw_frame, n_frames, Path(__file__).parent / 'damping_sweep.gif', fps=25)

plt.show()
