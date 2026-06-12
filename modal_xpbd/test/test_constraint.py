"""verify the hand-built constraint jacobians against autodiff of the residual,
differentiated through the exact same displace() used to apply solver updates
"""
import jax
from jax import numpy as jnp

from modal_xpbd.body import ModalBody
from modal_xpbd.constraint import modal_jacobian, pin, pin_world, residual, twist_jacobian
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.truss import girder

jax.config.update('jax_enable_x64', True)


def make_bodies():
	shape = reduce_modes(girder(4, stiffness=100.0), 5)
	b0 = ModalBody.rest(shape, angle=0.3, position=(0.0, 0.0))
	b1 = ModalBody.rest(shape, angle=-0.2, position=(4.2, 0.5))
	# nonzero amplitudes so the modal displacement enters the twist jacobian lever arm
	b0 = b0.replace(amplitudes=jnp.linspace(0.1, 0.5, 5))
	b1 = b1.replace(amplitudes=jnp.linspace(-0.3, 0.2, 5))
	return [b0, b1]


def autodiff_jacobians(c, bodies):
	k = [b.shape.n_modes for b in bodies]

	def res(d):
		moved = [b.displace(dt, dq) for b, (dt, dq) in zip(bodies, d)]
		return residual(c, moved)

	zero = [(jnp.zeros(3), jnp.zeros(ki)) for ki in k]
	return jax.jacobian(res)(zero)


def test_pin_jacobians():
	bodies = make_bodies()
	c = pin(bodies, 0, 1, world_point=(2.0, 0.3))
	J = autodiff_jacobians(c, bodies)

	assert jnp.allclose(J[0][0], twist_jacobian(bodies[0], c.anchor_a, c.modes_a, -1.0))
	assert jnp.allclose(J[0][1], modal_jacobian(bodies[0], c.modes_a, -1.0))
	assert jnp.allclose(J[1][0], twist_jacobian(bodies[1], c.anchor_b, c.modes_b, +1.0))
	assert jnp.allclose(J[1][1], modal_jacobian(bodies[1], c.modes_b, +1.0))


def test_pin_world_jacobians():
	bodies = make_bodies() + [ModalBody.world()]
	c = pin_world(bodies, 0, world_point=(-2.0, 0.0))
	J = autodiff_jacobians(c, bodies)

	assert jnp.allclose(J[0][0], twist_jacobian(bodies[0], c.anchor_a, c.modes_a, -1.0))
	assert jnp.allclose(J[0][1], modal_jacobian(bodies[0], c.modes_a, -1.0))
	assert jnp.allclose(J[1][0], 0)
	assert jnp.allclose(J[1][1], 0)
	# the world side carries an ordinary jacobian;
	# its zero inverse mass keeps the world in place
	assert jnp.allclose(J[2][0], twist_jacobian(bodies[2], c.anchor_b, c.modes_b, +1.0))
