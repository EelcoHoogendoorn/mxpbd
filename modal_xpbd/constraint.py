"""point constraints between modal bodies

On a flexible body there is no such thing as a lumped joint;
what physically exists is bearings at points.
Hinges, welds and splices all arise as (pairs of) point constraints,
which is why these are solved together as a block downstream.

Anchors are stored in body local coordinates,
together with the mode shapes sampled at the anchor.
The modal jacobian of a point constraint is just those sampled mode shapes,
rotated to the world frame: modal displacement is linear in the amplitudes.

Note that constraint pairs on near-rigid bodies are redundant:
a splice pair constrains 4 directions of which only 3 exist between rigid bodies;
the bolt-to-bolt distance is taken up by flex alone.
The solver regularizes the system against this (see solve.solve_constraints).
"""
import dataclasses

import jax
from jax import numpy as jnp

from modal_xpbd.body import ModalBody, perp
from modal_xpbd.decompose import find_vertex
from modal_xpbd.pytree import register, static_field

# index of the world body: by convention the last entry of the bodies list
# (ModalBody.world(); zero inverse mass, frame equal to the world frame),
# so plain negative indexing reaches it through every generic code path.
# attachment to the world is thereby an ordinary two-body constraint
WORLD = -1


@register
@dataclasses.dataclass
class PointConstraint:
	"""pin between anchor points on two bodies, or on one body and the world"""
	body_a: int = static_field()
	body_b: int = static_field()	# WORLD pins to the world body
	anchor_a: jax.Array = None		# [2] body local
	anchor_b: jax.Array = None		# [2] body local; on the world body that is the world point
	modes_a: jax.Array = None		# [ka, 2] mode shapes sampled at anchor_a
	modes_b: jax.Array = None		# [kb, 2]
	compliance: jax.Array = None	# scalar
	regularization: jax.Array = None	# scalar; small compliance floor keeping redundant pairs regular


def pin(bodies, a: int, b: int, world_point, compliance=0.0, regularization=1e-16) -> PointConstraint:
	"""pin the vertices of bodies[a] and bodies[b] nearest to world_point together"""
	def sample(i):
		body = bodies[i]
		local = body.rotation().T @ (jnp.asarray(world_point) * 1.0 - body.position)
		return body.shape.sample(find_vertex(body.shape, local))

	anchor_a, modes_a = sample(a)
	anchor_b, modes_b = sample(b)
	return PointConstraint(
		body_a=a, body_b=b,
		anchor_a=anchor_a, anchor_b=anchor_b,
		modes_a=modes_a, modes_b=modes_b,
		compliance=jnp.asarray(compliance) * 1.0,
		regularization=jnp.asarray(regularization) * 1.0,
	)


def pin_world(bodies, a: int, world_point, compliance=0.0, regularization=1e-16) -> PointConstraint:
	"""pin the vertex of bodies[a] nearest to world_point to that world point

	The world side is the world body: its frame is the world frame,
	so its anchor is simply the world point, with no modes behind it.
	"""
	assert float(bodies[WORLD].shape.mass_inv) == 0.0, \
		"the bodies list must end with the world body; append ModalBody.world()"
	body = bodies[a]
	local = body.rotation().T @ (jnp.asarray(world_point) * 1.0 - body.position)
	anchor_a, modes_a = body.shape.sample(find_vertex(body.shape, local))
	return PointConstraint(
		body_a=a, body_b=WORLD,
		anchor_a=anchor_a, anchor_b=jnp.asarray(world_point) * 1.0,
		modes_a=modes_a, modes_b=jnp.zeros((0, 2)),
		compliance=jnp.asarray(compliance) * 1.0,
		regularization=jnp.asarray(regularization) * 1.0,
	)


def localize(groups):
	"""restructure each constraint group onto the bodies its constraints reference

	Returns (support, local) per group:
	support: sorted indices of the bodies the group's constraints touch
	local: the group's constraints, re-indexed onto that support

	A group solve thereby scales with the size of the group, not the body count.
	Body indices are static metadata; this happens entirely at trace time.
	The world body, indexed WORLD = -1, sorts first into any support that pins to it,
	and joins the group solve like any other body.
	"""
	local_groups = []
	for group in groups:
		support = sorted({i for c in group for i in (c.body_a, c.body_b)})
		remap = {g: l for l, g in enumerate(support)}
		local = [dataclasses.replace(
			c,
			body_a=remap[c.body_a],
			body_b=remap[c.body_b],
		) for c in group]
		local_groups.append((support, local))
	return local_groups


def anchor_world(body: ModalBody, anchor, modes):
	"""world position of an anchor, including its modal displacement"""
	return body.rotation() @ (anchor + modes.T @ body.amplitudes) + body.position


def residual(constraint: PointConstraint, bodies):
	"""[2] world space separation of the two anchor points"""
	c = constraint
	pa = anchor_world(bodies[c.body_a], c.anchor_a, c.modes_a)
	pb = anchor_world(bodies[c.body_b], c.anchor_b, c.modes_b)
	return pb - pa


def twist_jacobian(body: ModalBody, anchor, modes):
	"""[2, 3] derivative of the world anchor position to the body twist (angular, linear, linear)"""
	r = body.rotation() @ (anchor + modes.T @ body.amplitudes)
	return jnp.concatenate([perp(r)[:, None], jnp.eye(2)], axis=1)


def modal_jacobian(body: ModalBody, modes):
	"""[2, n_modes] derivative of the world anchor position to the modal amplitudes;
	simply the sampled mode shapes, rotated to the world frame"""
	return body.rotation() @ modes.T
