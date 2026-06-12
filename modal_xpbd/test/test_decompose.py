import jax
import numpy as np
from jax import numpy as jnp

from modal_xpbd.body import perp
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.truss import girder

jax.config.update('jax_enable_x64', True)


def test_girder_assembly():
	t = girder(4)
	K = t.stiffness_matrix()
	assert np.allclose(K, K.T)
	# rigid translations and rotation about any point are in the nullspace
	n = t.n_nodes
	tx = np.tile([1.0, 0.0], n)
	ty = np.tile([0.0, 1.0], n)
	rot = np.stack([-t.vertices[:, 1], t.vertices[:, 0]], axis=1).flatten()
	for v in [tx, ty, rot]:
		assert np.allclose(K @ v, 0, atol=1e-9)


def test_reduce():
	t = girder(6, stiffness=100.0)
	n_modes = 8
	s = reduce_modes(t, n_modes)

	assert s.n_modes == n_modes
	assert (s.omega > 0).all()
	assert (jnp.diff(s.omega) >= 0).all()

	# mass-orthonormal modes
	gram = jnp.einsum('knd,n,jnd->kj', s.modes, s.masses, s.modes)
	assert jnp.allclose(gram, jnp.eye(n_modes), atol=1e-9)

	# orthogonal to the rigid modes: no net momentum in any mode
	linear = jnp.einsum('knd,n->kd', s.modes, s.masses)
	angular = jnp.einsum('knd,n,nd->k', s.modes, s.masses, perp(s.vertices))
	assert jnp.allclose(linear, 0, atol=1e-9)
	assert jnp.allclose(angular, 0, atol=1e-9)

	# com-centered
	assert jnp.allclose((s.vertices * s.masses[:, None]).sum(axis=0), 0, atol=1e-9)
