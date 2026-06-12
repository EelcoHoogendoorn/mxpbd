"""visual test: settled cantilever sag against the full order fem reference

The truncation gap of the free-free modal basis is directly visible:
low mode counts under-sag, the full basis lands on the fem shape.

Blocking plot test; under a non-interactive backend plt.show() is a no-op,
so this still runs headless as a smoke test of the plotting path.
"""
import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from matplotlib.collections import LineCollection

from modal_xpbd.body import ModalBody
from modal_xpbd.constraint import pin_world
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.solve import step
from modal_xpbd.truss import girder

jax.config.update('jax_enable_x64', True)

step_jit = jax.jit(step, static_argnames=('substeps',))


def test_cantilever_sag_visual():
	truss = girder(8, stiffness=1e4)
	gravity = (0.0, -0.5)
	reference = truss.solve_static(fixed_nodes=[0, 9], force=truss.gravity_force(gravity))

	def settle(n_modes):
		shape = reduce_modes(truss, n_modes)
		bodies = [ModalBody.rest(shape), ModalBody.world()]
		constraints = [
			pin_world(bodies, 0, world_point=np.asarray(shape.vertices[0])),
			pin_world(bodies, 0, world_point=np.asarray(shape.vertices[9])),
		]
		for i in range(1000):
			bodies = step_jit(bodies, [constraints], dt=0.05, substeps=2, gravity=gravity, damping=0.4)
		return bodies[0]

	settled = {k: settle(k) for k in [4, 12, 33]}
	assert all(jnp.isfinite(b.position).all() for b in settled.values())

	fig, ax = plt.subplots(figsize=(12, 5))
	rest = np.asarray(settled[33].shape.vertices)	# com-centered undeformed shape
	edges = np.asarray(truss.edges)

	ax.add_collection(LineCollection(rest[edges], colors='lightgray', linewidths=0.8, label='undeformed'))
	for (k, body), color in zip(settled.items(), ['tab:orange', 'tab:green', 'tab:blue']):
		points = np.asarray(body.world_points())
		ax.add_collection(LineCollection(points[edges], colors=color, linewidths=0.8, label=f'{k} modes'))
	# drawn last; with the full basis it should disappear into the 33 mode shape
	ax.add_collection(LineCollection(
		(rest + reference)[edges], colors='crimson', linewidths=1.4, linestyles=':', label='full order fem'))

	ax.autoscale()
	ax.set_aspect('equal')
	ax.legend(loc='lower left')
	ax.set_title('cantilever girder settled under gravity; free-free modal basis vs full order fem')
	plt.close()
