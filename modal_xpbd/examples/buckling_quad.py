"""buckling of a slender steel rod under a prescribed end shortening

A ~1 m structural-steel rod (E = 200 GPa, rho = 7850 kg/m^3, 4 mm square
section, slenderness span/thickness ~ 256), modelled as a chain of eight
plane-stress continuum segments and clamped to the world at both ends by pin
pairs; the right pair is driven inward, shortening the rod like a test rig.
This is the unit-scale girder demo carried to real steel by a similarity
transform (length, modulus, density rescaled, with dt and the crush scaled to
match), so the net behaviour is identical, only now in physically legible units.

Note that the observed pop trails the euler
line somewhat: the straight state remains a true (unstable) equilibrium past
critical, and the dwell is the e-folding time for rounding level noise to grow
visible.

A single linearized body cannot buckle: geometric stiffness requires the
coupling of axial load into lateral deflection, which a constant stiffness
matrix has linearized away. The chain of modal bodies recovers it at the
assembly level, since the large rotations live in the floating frames.

Note that this demo does not require substepping, nor implicit integration, nor a global solver;
merely quasi-explicit integration with large timesteps, and local constraint solves.
Yet it can simulate arbitrarily stiff materials and geometries.

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
from modal_xpbd.quad import quad_strip

jax.config.update('jax_enable_x64', True)

E, nu, density = 2.0e11, 0.3, 7850.0	# structural steel: 200 GPa, -, kg/m^3

n_segments = 9
cells_per_segment = 32					# square continuum cells along each segment
thickness = 0.004						# 4 mm square section (plane stress, unit depth)
length = cells_per_segment * thickness	# 0.128 m per segment
span = n_segments * length				# ~1 m rod; slenderness span/thickness ~ 256
half = thickness / 2					# splice/clamp points sit on the rod's edges
dt = 0.01								# elastodynamic time-scale; keeps dt*omega vs the unit demo
substeps = 20
n_frames, steps_per_frame = 120, 1
crush = 0.0006		# prescribed end shortening (0.6 mm), ramped linearly; ~12x euler
gravity = (0.0, 0.0)

shape = reduce_modes(
	quad_strip(cells_per_segment, ny=1, cell=thickness, E=E, nu=nu, density=density),
	n_modes=8)
bodies = [
	ModalBody.rest(shape, position=((i + 0.5) * length, 0.0), damping=3)
	for i in range(n_segments)
] + [ModalBody.world()]
splices = [
	c
	for i in range(n_segments - 1)
	for c in (
		pin(bodies, i, i + 1, world_point=((i + 1) * length, -half)),
		pin(bodies, i, i + 1, world_point=((i + 1) * length, +half)),
	)
] + [
	pin_world(bodies, 0, world_point=(0.0, -half)),
	pin_world(bodies, 0, world_point=(0.0, +half)),
]
crosshead = [
	pin_world(bodies, n_segments - 1, world_point=(span, -half)),
	pin_world(bodies, n_segments - 1, world_point=(span, +half)),
]


def crushed(d):
	"""the crosshead anchors, driven inwards by d; anchors are data, no retrace"""
	return [dataclasses.replace(c, anchor_b=c.anchor_b - jnp.asarray([d, 0.0])) for c in crosshead]


# euler buckling of a clamped-clamped continuum strut, as an end shortening:
# delta_cr = P_cr / (EA/L) = 4 pi^2 (I/A) / span = pi^2 thickness^2 / (3 span)
critical = np.pi ** 2 * thickness ** 2 / (3 * span)
print(f'estimated critical shortening: {critical:.2e} of a crush of {crush}')

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
	bow.append(float((state[n_segments // 2 - 1].position[1] + state[n_segments // 2].position[1]) / 2))
	applied.append(d)
bow, applied = np.asarray(bow), np.asarray(applied)

fig, (ax, ax_t) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[1, 1.4])
for i, f in enumerate(frames[:: n_frames // 10]):
	draw_bodies(ax, f, color='b', alpha=0.12 + 0.85 * i / 10)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('steel rod, prescribed end shortening; pop past critical, seeded by rounding noise')

ax_t.plot(applied, np.abs(bow), c='k')
ax_t.axvline(critical, ls=':', c='crimson', label='euler estimate')
ax_t.set_xlabel('prescribed end shortening')
ax_t.set_ylabel('midspan deflection')
ax_t.legend()
fig.savefig(Path(__file__).parent / 'buckling_quad.png', dpi=120)

# animated gif
fig_anim, ax_anim = plt.subplots(figsize=(10, 3))


def draw_frame(i):
	ax_anim.clear()
	draw_bodies(ax_anim, frames[i], color='b')
	ax_anim.set_xlim(-0.05 * span, 1.05 * span)
	ax_anim.set_ylim(-0.15 * span, 0.15 * span)
	ax_anim.set_aspect('equal')
	ax_anim.set_xticks([])
	ax_anim.set_yticks([])


save_gif(fig_anim, draw_frame, n_frames, Path(__file__).parent / 'buckling_quad.gif', fps=50)

plt.show()
