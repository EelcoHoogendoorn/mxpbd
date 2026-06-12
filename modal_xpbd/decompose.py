"""modal reduction of a truss into a small set of vibration modes

Free-free eigenmodes of the unconstrained structure;
the three rigid modes are dropped and carried as explicit se2 degrees of freedom instead.

Modes are mass-orthonormal, so downstream the modal mass matrix is identity
and the modal stiffness is omega**2 per mode.
"""
import dataclasses

import jax
import numpy as np
import scipy.linalg
from jax import numpy as jnp

from modal_xpbd.pytree import register
from modal_xpbd.truss import Truss


@register
@dataclasses.dataclass
class ReducedShape:
	"""modally reduced 2d body, described about its center of mass"""
	vertices: jax.Array		# [n, 2] com-centered rest positions
	edges: jax.Array		# [e, 2] for rendering
	masses: jax.Array		# [n] lumped nodal masses
	modes: jax.Array		# [n_modes, n, 2] mass-orthonormal mode shapes
	omega: jax.Array		# [n_modes] angular frequencies
	compliance: jax.Array	# [n_modes] modal compliance; 1 / omega**2 for the elastic body
	mass: jax.Array			# scalar; total mass
	inertia: jax.Array		# scalar; rotational inertia about the com
	mass_inv: jax.Array		# scalar; stored inverses, zero for the world body
	inertia_inv: jax.Array	# scalar

	@property
	def n_modes(self):
		return self.modes.shape[0]

	def sample(self, idx):
		"""rest position and mode shapes at vertex idx: ([2], [n_modes, 2])"""
		return self.vertices[idx], self.modes[:, idx]

	def rigid(self) -> "ReducedShape":
		"""the same shape with incompliant modes: an exactly rigid body,
		that still carries its modal dofs through the same code paths"""
		return dataclasses.replace(self, compliance=self.compliance * 0)


def find_vertex(shape: ReducedShape, point):
	"""index of the vertex nearest to point, in body local coordinates"""
	return int(jnp.argmin(((shape.vertices - jnp.asarray(point)) ** 2).sum(axis=1)))


def reduce_modes(truss: Truss, n_modes: int, rigid_tol=1e-6) -> ReducedShape:
	"""reduce a truss to its n_modes lowest free-free vibration modes"""
	K = truss.stiffness_matrix()
	m = truss.masses()
	M = np.repeat(m, 2)

	w2, phi = scipy.linalg.eigh(K, np.diag(M))
	# free-free structure: exactly three rigid zero modes, carried as se2 dofs instead
	assert w2[3] > 0
	assert abs(w2[2]) < w2[3] * rigid_tol, "structure has internal mechanisms; brace it"
	w2, phi = w2[3:3 + n_modes], phi[:, 3:3 + n_modes]

	com = (truss.vertices * m[:, None]).sum(axis=0) / m.sum()
	vertices = truss.vertices - com
	mass = m.sum()
	inertia = (m * (vertices ** 2).sum(axis=1)).sum()
	return ReducedShape(
		vertices=jnp.asarray(vertices),
		edges=jnp.asarray(truss.edges),
		masses=jnp.asarray(m),
		modes=jnp.asarray(phi.T.reshape(n_modes, truss.n_nodes, 2)),
		omega=jnp.asarray(np.sqrt(w2)),
		compliance=jnp.asarray(1 / w2),
		mass=jnp.asarray(mass),
		inertia=jnp.asarray(inertia),
		mass_inv=jnp.asarray(1 / mass),
		inertia_inv=jnp.asarray(1 / inertia),
	)
