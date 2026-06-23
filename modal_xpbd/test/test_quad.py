"""the Q4 continuum mesh as a drop-in for the modal pipeline

The truss is a lattice; this checks that a genuine plane-stress continuum
assembles a well-posed stiffness (symmetric, exactly three rigid modes, no
spurious mechanisms) and that a free-floating reduced body behaves physically:
it rings on its modes at the eigenfrequency while its rigid frame stays put.
"""
import jax
import numpy as np
from jax import numpy as jnp

from modal_xpbd.body import ModalBody
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.quad import quad_strip
from modal_xpbd.solve import step

jax.config.update('jax_enable_x64', True)

step_jit = jax.jit(step, static_argnames=('substeps',))


def test_assembly_is_well_posed():
	"""symmetric stiffness with exactly three rigid-body modes and no hourglass"""
	mesh = quad_strip(nx=6, ny=2, cell=0.5, E=100.0, nu=0.3, density=1.0)
	K = mesh.stiffness_matrix()
	assert np.allclose(K, K.T)
	w = np.linalg.eigvalsh(K)
	# three near-zero rigid modes (2 translation, 1 rotation), then a real gap:
	# full integration leaves no extra zero-energy (hourglass) modes
	assert np.abs(w[:3]).max() < 1e-8 * w[-1]
	assert w[3] > 1e-6 * w[-1]


def test_free_body_rings_and_conserves_momentum():
	"""a kicked free-floating quad strip oscillates at its eigenfrequency,
	while the rigid frame never moves: the modes carry no net momentum"""
	mesh = quad_strip(nx=8, ny=2, cell=0.5, E=100.0, nu=0.3, density=1.0)
	shape = reduce_modes(mesh, n_modes=6)
	omega = float(shape.omega[0])

	bodies = [ModalBody.rest(shape).replace(rates=jnp.zeros(6).at[0].set(1.0))]
	dt = 2 * np.pi / omega / 60
	trace = []
	for i in range(240):
		bodies = step_jit(bodies, [], dt=dt, substeps=4)
		trace.append(float(bodies[0].amplitudes[0]))
	trace = np.asarray(trace)

	# the rate kick swings into amplitude ~ 1 / omega
	assert np.abs(trace).max() > 0.5 / omega
	# at the eigenfrequency: period measured from upward zero crossings
	sign = np.sign(trace)
	up = np.nonzero((sign[1:] > 0) & (sign[:-1] <= 0))[0]
	period = 2 * np.pi / omega
	assert abs(np.diff(up).mean() * dt - period) < period * 0.05
	# the frame stays put: flex modes are momentum-free
	assert jnp.allclose(bodies[0].position, 0, atol=1e-9)
	assert jnp.allclose(bodies[0].angle, 0, atol=1e-9)
