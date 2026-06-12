"""a chain of girders swinging from a single pin, released from horizontal

Joints here are single point constraints: hinges, transmitting force but no moment,
in contrast to the splice pairs of the bridge example. The chain is an n-pendulum
with flexible links, swinging through arbitrarily large rotations:
the large angles live in the floating frames, at no cost to the linear modal flex,
which remains valid within each body throughout the whip.

The energy trace is the correctness exhibit for the large angle regime:
monotone decay toward the hanging rest energy through the prescribed modal damping,
dissipating in bursts where the whip excites flex,
with no spurious gain at any rotation.
The damping acts on the flex alone: the chain coasts undamped through the
smooth swing phases; there is no global velocity drag of the kind rigid body
engines employ.

The example works equally well and efficient over a large stiffness range;
from perfectly rigid bodies, all the way to soft bodies,
as long as their modal amplitudes remain reasonable.
To demonstrate, the links span that range within a single chain:
exactly rigid at the pin (zero compliance, through the same code path),
then softening by a factor of four per link toward the tip,
where the whip makes the flex most visible.
Bodies of different compliance are nothing special: every body carries
its own shape, and the solver is indifferent to the mix.

If large deformable jello like behavior is required, the modal reduction is not appropriate,
and a large number of DOFs are inherently required to capture that dynamic;
but accurately and efficiently modelling the flex of real engineering solids,
is where this method shines.

"""
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from modal_xpbd.body import ModalBody
from modal_xpbd.constraint import pin, pin_world, residual
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.plot import draw_bodies, save_gif
from modal_xpbd.solve import step
from modal_xpbd.truss import girder

jax.config.update('jax_enable_x64', True)

n_girders = 4
n_cells = 4
stiffness = 6e4
length = float(n_cells)
reach = n_girders * length
gravity = (0.0, -2.0)
dt = 0.05
n_frames, steps_per_frame = 400, 2
damping = 0.02

# one shape per link, an order of magnitude more compliant than the last;
# the first is exactly rigid: the zero compliance end of the range
shapes = [
	reduce_modes(girder(n_cells, stiffness=stiffness / 4 ** i), n_modes=6)
	for i in range(n_girders)
]
shapes[0] = shapes[0].rigid()
colors = ['dimgray'] + [f'C{i}' for i in range(n_girders - 1)]
labels = ['rigid'] + [f'EA = {stiffness / 4 ** i:.0e}' for i in range(1, n_girders)]
bodies = [
	ModalBody.rest(shape, position=((i + 0.5) * length, 0.0))
	for i, shape in enumerate(shapes)
] + [ModalBody.world()]
# single pins: hinges at the top corner of each interface, and at the support
hinges = [
	pin(bodies, i, i + 1, world_point=((i + 1) * length, 0.5))
	for i in range(n_girders - 1)
] + [
	pin_world(bodies, 0, world_point=(0.0, 0.5)),
]

step_jit = jax.jit(step, static_argnames=('substeps',))

frames, energy, violation = [], [], []
state = bodies
for f in range(n_frames):
	for i in range(steps_per_frame):
		state = step_jit(state, [hinges], dt=dt, substeps=4, gravity=gravity, damping=damping)
	frames.append(state)
	energy.append(sum(float(b.energy(gravity)) for b in state))
	violation.append(max(float(jnp.linalg.norm(residual(c, state))) for c in hinges))
energy, violation = np.asarray(energy), np.asarray(violation)

fig, (ax, ax_e) = plt.subplots(2, 1, figsize=(9, 9), height_ratios=[2, 1])
for i, f in enumerate(frames[:: n_frames // 12]):
	for body, color in zip(f, colors):
		draw_bodies(ax, [body], color=color, alpha=0.1 + 0.85 * i / 12)
ax.autoscale()
ax.set_aspect('equal')
ax.legend(handles=[plt.Line2D([], [], color=c, label=l) for c, l in zip(colors, labels)])
ax.set_title(f'hinged chain released from horizontal; max pin violation {violation.max():.1e}')

t = np.arange(n_frames) * dt * steps_per_frame
ax_e.plot(t, energy, c='k')
# the physical floor: the chain hanging straight down, at rest
floor = sum(
	-float(shape.mass) * gravity[1] * (0.5 - (i + 0.5) * length)
	for i, shape in enumerate(shapes))
ax_e.axhline(floor, ls=':', c='crimson', label='hanging at rest')
ax_e.set_xlabel('time')
ax_e.set_ylabel('total energy')
ax_e.legend()
fig.savefig(Path(__file__).parent / 'swinging_chain.png', dpi=120)

# animated gif
fig_anim, ax_anim = plt.subplots(figsize=(6, 6))


def draw_frame(i):
	ax_anim.clear()
	for body, color in zip(frames[i], colors):
		draw_bodies(ax_anim, [body], color=color)
	ax_anim.set_xlim(-reach - 1, reach + 1)
	ax_anim.set_ylim(-reach - 1, reach * 0.3)
	ax_anim.set_aspect('equal')
	ax_anim.set_xticks([])
	ax_anim.set_yticks([])


save_gif(fig_anim, draw_frame, n_frames, Path(__file__).parent / 'swinging_chain.gif', fps=25)

plt.show()
