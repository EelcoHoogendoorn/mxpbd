"""rigid + modal body state

In 2d, rigid poses are (angle, position) and twists are (angular_velocity, velocity);
rotation is abelian, so no exponential map machinery is required.

With the body origin at the center of mass and mass-orthonormal modes,
the mass matrix in (twist, modal) coordinates is constant diagonal:
(inertia, mass, mass, 1, ..., 1).
This holds to first order in the modal amplitudes,
consistent with the linear modes themselves.
"""
import dataclasses

import jax
from jax import numpy as jnp

from modal_xpbd.decompose import ReducedShape
from modal_xpbd.pytree import register


def rotation(angle):
	"""[2, 2] rotation matrix"""
	c, s = jnp.cos(angle), jnp.sin(angle)
	return jnp.array([[c, -s], [s, c]])


def perp(v):
	"""90 degree counterclockwise rotation; the derivative of rotation with respect to angle"""
	return jnp.stack([-v[..., 1], v[..., 0]], axis=-1)


@register
@dataclasses.dataclass
class ModalBody:
	shape: ReducedShape
	angle: jax.Array			# scalar
	position: jax.Array			# [2] com world position
	amplitudes: jax.Array		# [n_modes] modal amplitudes
	angular_velocity: jax.Array	# scalar
	velocity: jax.Array			# [2] com world velocity
	rates: jax.Array			# [n_modes] modal amplitude rates

	@classmethod
	def world(cls) -> "ModalBody":
		"""the world as a body: zero inverse mass, no modes, nothing to draw

		Constraints anchor to the world by pinning to this body (see constraint.WORLD).
		Its frame is the world frame, and zero inverse mass keeps it static under
		any force, so the solve needs no special case for it.
		"""
		shape = ReducedShape(
			vertices=jnp.zeros((0, 2)),
			edges=jnp.zeros((0, 2), dtype=int),
			masses=jnp.zeros(0),
			modes=jnp.zeros((0, 0, 2)),
			omega=jnp.zeros(0),
			compliance=jnp.zeros(0),
			mass=jnp.zeros(()),
			inertia=jnp.zeros(()),
			mass_inv=jnp.zeros(()),
			inertia_inv=jnp.zeros(()),
		)
		return cls.rest(shape)

	@classmethod
	def rest(cls, shape: ReducedShape, angle=0.0, position=(0.0, 0.0)) -> "ModalBody":
		k = shape.n_modes
		return cls(
			shape=shape,
			angle=jnp.asarray(angle) * 1.0,
			position=jnp.asarray(position) * 1.0,
			amplitudes=jnp.zeros(k),
			angular_velocity=jnp.zeros(()),
			velocity=jnp.zeros(2),
			rates=jnp.zeros(k),
		)

	def replace(self, **kwargs) -> "ModalBody":
		return dataclasses.replace(self, **kwargs)

	def rotation(self):
		return rotation(self.angle)

	def twist_mass_inv(self):
		"""[3] diagonal inverse mass in (angular, linear, linear) twist coordinates"""
		s = self.shape
		return jnp.stack([s.inertia_inv, s.mass_inv, s.mass_inv])

	def local_points(self):
		"""[n, 2] vertex positions in body frame, including modal displacement"""
		s = self.shape
		return s.vertices + jnp.einsum('k,knd->nd', self.amplitudes, s.modes)

	def world_points(self):
		"""[n, 2] vertex positions in world frame"""
		return self.local_points() @ self.rotation().T + self.position

	def displace(self, d_twist, d_amplitudes) -> "ModalBody":
		"""apply a position-level displacement in (twist, modal) coordinates"""
		return self.replace(
			angle=self.angle + d_twist[0],
			position=self.position + d_twist[1:],
			amplitudes=self.amplitudes + d_amplitudes,
		)

	def energy(self, gravity=(0.0, 0.0)):
		"""total mechanical energy; modal mass is identity by construction"""
		s = self.shape
		kinetic = (
			s.inertia * self.angular_velocity ** 2 +
			s.mass * (self.velocity ** 2).sum() +
			(self.rates ** 2).sum()
		) / 2
		elastic = ((s.omega * self.amplitudes) ** 2).sum() / 2
		potential = -s.mass * jnp.asarray(gravity) @ self.position
		return kinetic + elastic + potential
