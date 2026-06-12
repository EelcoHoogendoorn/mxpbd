"""end to end dynamics tests

The girder chain exercises the central claims:
splice joints as solved-together pairs of point constraints,
stability under substepping (the position-velocity update),
and quantitative agreement of the reduced modal model with the full order truss.
"""
import jax
import numpy as np
from jax import numpy as jnp

from modal_xpbd.body import ModalBody
from modal_xpbd.constraint import pin, pin_world, residual
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.solve import step
from modal_xpbd.truss import girder

jax.config.update('jax_enable_x64', True)

step_jit = jax.jit(step, static_argnames=('substeps',))


def spliced_pair(stiffness=100.0, n_modes=6):
	"""two girders joined end to end by a splice: a pair of point constraints"""
	shape = reduce_modes(girder(4, stiffness=stiffness), n_modes)
	bodies = [
		ModalBody.rest(shape, position=(2.0, 0.5)),
		ModalBody.rest(shape, position=(6.0, 0.5)),
		ModalBody.world(),
	]
	splice = [
		pin(bodies, 0, 1, world_point=(4.0, 0.0)),
		pin(bodies, 0, 1, world_point=(4.0, 1.0)),
	]
	return bodies, splice


def test_pin_convergence():
	"""a violated constraint is projected out within a few steps"""
	bodies, splice = spliced_pair()
	# misplace the second body so the splice starts out violated
	bodies[1] = bodies[1].replace(position=jnp.asarray([6.2, 0.6]), angle=jnp.asarray(0.1))

	violation = [jnp.linalg.norm(jnp.concatenate([residual(c, bodies) for c in splice]))]
	for i in range(5):
		bodies = step_jit(bodies, [splice], dt=0.05, substeps=1, damping=0.5)
		violation.append(jnp.linalg.norm(jnp.concatenate([residual(c, bodies) for c in splice])))

	assert violation[-1] < violation[0] * 1e-3


def test_substep_energy_stable():
	"""undamped ring-down does not gain energy, for low and high substep counts alike"""
	bodies, splice = spliced_pair()
	constraints = splice + [pin_world(bodies, 0, world_point=(0.0, 0.0))]
	# transverse kick on the far body; swinging chain with flex ringing
	bodies[1] = bodies[1].replace(velocity=jnp.asarray([0.0, 0.3]))

	def energy(bodies):
		return sum(b.energy() for b in bodies)

	e0 = energy(bodies)
	for substeps in [1, 8]:
		trajectory = [b for b in bodies]
		energies = []
		for i in range(250):
			trajectory = step_jit(trajectory, [constraints], dt=0.02, substeps=substeps)
			energies.append(energy(trajectory))
		energies = jnp.asarray(energies)
		assert jnp.all(jnp.isfinite(energies))
		assert jnp.max(energies) < e0 * 1.05		# no energy gain; substepping is stable
		assert energies[-1] > e0 * 0.5				# nor is it spuriously dissipative


def test_rigid_modes():
	"""a shape with incompliant modes is exactly a rigid body

	The same modal dofs ride through the same code paths, pinned at zero.
	The splice pair is redundant between rigid bodies (the bolt-to-bolt distance
	has no flex to load); the solver regularization keeps the system regular.
	"""
	shape = reduce_modes(girder(4, stiffness=1e3), 6).rigid()
	bodies = [ModalBody.rest(shape, position=((i + 0.5) * 4.0, 0.5)) for i in range(2)] + [ModalBody.world()]
	groups = [[
		pin(bodies, 0, 1, world_point=(4.0, 0.0)),
		pin(bodies, 0, 1, world_point=(4.0, 1.0)),
		pin_world(bodies, 0, world_point=(0.0, 0.0)),
	]]
	for substeps in [1, 8]:
		state = bodies
		for i in range(50):
			state = step_jit(state, groups, dt=0.05, substeps=substeps, gravity=(0.0, -1.0))
		assert all(jnp.isfinite(b.position).all() for b in state)
		assert all(jnp.allclose(b.amplitudes, 0, atol=1e-9) for b in state)


def test_unresolved_modes():
	"""modes far above the substep rate are implicitly low-passed, not unstable

	One substep acts on each mode as implicit euler: the phase plane radius
	contracts by exactly 1 / sqrt(1 + (omega h)**2) per substep.
	Static compliance is preserved regardless: the settled deflection
	is independent of the timestep, however unresolved the upper modes.
	"""
	shape = reduce_modes(girder(4, stiffness=1e4), 8)

	# free decay of the stiffest mode at omega h = 10
	k = shape.n_modes - 1
	omega = float(shape.omega[k])
	h = 10.0 / omega
	bodies = [ModalBody.rest(shape).replace(amplitudes=jnp.zeros(8).at[k].set(1.0))]

	def radius(body):
		return jnp.sqrt(body.amplitudes[k] ** 2 + (body.rates[k] / omega) ** 2)

	r = [radius(bodies[0])]
	for i in range(5):
		bodies = step_jit(bodies, [], dt=h, substeps=1)
		r.append(radius(bodies[0]))
	r = jnp.asarray(r)
	assert jnp.allclose(r[1:] / r[:-1], 1 / jnp.sqrt(1 + (omega * h) ** 2), rtol=1e-9)

	# quasi-static deformability survives: settled sag is timestep independent,
	# with the upper modes unresolved at either timestep
	gravity = (0.0, -0.5)

	def settle(dt, n):
		bodies = [ModalBody.rest(shape), ModalBody.world()]
		groups = [[
			pin_world(bodies, 0, world_point=np.asarray(shape.vertices[0])),
			pin_world(bodies, 0, world_point=np.asarray(shape.vertices[5])),
		]]
		for i in range(n):
			bodies = step_jit(bodies, groups, dt=dt, substeps=1, gravity=gravity, damping=0.4)
		return float(bodies[0].world_points()[4, 1] - shape.vertices[4, 1])

	sag_fine, sag_coarse = settle(0.02, 800), settle(0.1, 300)
	assert sag_fine < -1e-3
	assert abs(sag_coarse - sag_fine) < abs(sag_fine) * 1e-3


def test_bridge_matches_nonlinear_fem():
	"""the girder chain captures assembly level geometric nonlinearity

	A truss span supported at its bottom corners softens geometrically:
	end rotation about the eccentric supports (below the neutral axis) feeds
	axial compression into the sagging span, the beam-column effect.
	Linear fem cannot see this; at this deflection it errs by ~half.
	The chain of strictly linear modal bodies tracks the geometrically exact
	equilibrium instead, to within its modal truncation bias,
	because the large rotations live in the floating frames.
	"""
	n_girders, n_cells, length = 4, 4, 4.0
	span = n_girders * n_cells
	gravity = (0.0, -1.0)

	truss = girder(span, stiffness=1e4)
	force = truss.gravity_force(gravity)
	linear = truss.solve_static([0, span], force)[span // 2, 1]
	nonlinear = truss.solve_static_nonlinear([0, span], force)[span // 2, 1]
	assert abs(nonlinear) > abs(linear) * 1.5	# well into the softening regime

	shape = reduce_modes(girder(n_cells, stiffness=1e4), 8)
	bodies = [ModalBody.rest(shape, position=((i + 0.5) * length, 0.5)) for i in range(n_girders)] + [ModalBody.world()]
	groups = [[
		c for i in range(n_girders - 1) for c in (
			pin(bodies, i, i + 1, world_point=((i + 1) * length, 0.0)),
			pin(bodies, i, i + 1, world_point=((i + 1) * length, 1.0)))
	] + [
		pin_world(bodies, 0, world_point=(0.0, 0.0)),
		pin_world(bodies, n_girders - 1, world_point=(float(span), 0.0)),
	]]
	for i in range(3500):
		bodies = step_jit(bodies, groups, dt=0.05, substeps=2, gravity=gravity, damping=0.5)
	sag = float(bodies[n_girders // 2 - 1].world_points()[n_cells, 1])

	assert abs(sag - nonlinear) < abs(nonlinear) * 0.2		# truncation-level agreement
	assert abs(sag - nonlinear) < abs(sag - linear)			# and unambiguously nonlinear


def test_groups_track_block_solve():
	"""per-splice gauss-seidel groups track the single exact block solve

	With the modal lambdas accumulated on the right hand side, grouping is a
	convergence knob rather than a correctness one: substepping reclaims the
	coupling error of smaller groups. They cannot agree exactly:
	gauss-seidel re-linearizes between groups, the single block solve does not.
	"""
	shape = reduce_modes(girder(4, stiffness=1e3), 6)

	def simulate(grouping):
		bodies = [ModalBody.rest(shape, position=((i + 0.5) * 4.0, 0.5)) for i in range(3)] + [ModalBody.world()]
		s0 = [pin(bodies, 0, 1, world_point=(4.0, 0.0)), pin(bodies, 0, 1, world_point=(4.0, 1.0))]
		s1 = [pin(bodies, 1, 2, world_point=(8.0, 0.0)), pin(bodies, 1, 2, world_point=(8.0, 1.0))]
		w = [pin_world(bodies, 0, world_point=(0.0, 0.0))]
		groups = {'single': [s0 + s1 + w], 'split': [s0, s1, w]}[grouping]
		for i in range(40):
			bodies = step_jit(
				bodies, groups, dt=0.05, substeps=2,
				gravity=(0.0, -1.0), damping=0.2)
		return np.concatenate([np.asarray(b.position) for b in bodies])

	distance = np.linalg.norm(simulate('split') - simulate('single'))
	assert distance < 1e-2


def test_free_vibration():
	"""a free floating body with kicked modal rates oscillates at its eigenfrequency

	Each mode is both a dof and an elastic constraint;
	the xpbd projection onto the modal constraint, plus the position-velocity update,
	is the oscillator. No point constraints or extra machinery involved.
	"""
	shape = reduce_modes(girder(4, stiffness=100.0), 6)
	omega = float(shape.omega[0])
	period = 2 * np.pi / omega
	bodies = [ModalBody.rest(shape).replace(rates=jnp.zeros(6).at[0].set(1.0))]

	dt = period / 100
	energy0 = float(bodies[0].energy())
	trace, energies = [], []
	for i in range(500):
		bodies = step_jit(bodies, [], dt=dt, substeps=1)
		trace.append(bodies[0].amplitudes[0])
		energies.append(bodies[0].energy())
	trace, energies = np.asarray(trace), np.asarray(energies)

	# it oscillates, swinging the kick into amplitude rate / omega
	assert abs(trace).max() > 0.9 / omega
	# at the eigenfrequency: measure the period from upward zero crossings
	sign = np.sign(trace)
	up = np.nonzero((sign[1:] > 0) & (sign[:-1] <= 0))[0]
	assert abs(np.diff(up).mean() * dt - period) < period * 0.02
	# the projection is implicit-euler flavored: it dissipates numerically,
	# at the implicit-euler rate of (omega h)**2 log energy per step.
	# substepping reduces the loss per simulated second linearly (small steps)
	loss = np.log(energy0 / energies[-1])
	assert abs(loss - 500 * (omega * dt) ** 2) < loss * 0.1
	# and the frame never moves: mass-orthonormal modes carry no momentum
	assert jnp.allclose(bodies[0].position, 0) and jnp.allclose(bodies[0].angle, 0)


def test_cantilever_sag_matches_fem():
	"""static sag of a world-pinned girder converges to the full order fem solution"""
	truss = girder(8, stiffness=1e4)
	gravity = (0.0, -0.5)
	tip = 8		# bottom right node
	reference = truss.solve_static(fixed_nodes=[0, 9], force=truss.gravity_force(gravity))
	assert reference[tip, 1] < -1e-3	# the reference itself meaningfully deflects

	def settled_sag_error(n_modes):
		shape = reduce_modes(truss, n_modes)
		bodies = [ModalBody.rest(shape), ModalBody.world()]
		# clamp the left edge: a pair of point constraints, not a joint abstraction
		constraints = [
			pin_world(bodies, 0, world_point=np.asarray(shape.vertices[0])),
			pin_world(bodies, 0, world_point=np.asarray(shape.vertices[9])),
		]
		for i in range(1000):
			bodies = step_jit(bodies, [constraints], dt=0.05, substeps=2, gravity=gravity, damping=0.4)
		assert jnp.linalg.norm(bodies[0].velocity) < 5e-3
		sag = bodies[0].world_points() - shape.vertices
		return abs(sag[tip, 1] - reference[tip, 1]) / abs(reference[tip, 1])

	error = {k: settled_sag_error(k) for k in [4, 12, 33]}
	# with the full flexible basis the projection reproduces the full order solution
	assert error[33] < 0.02
	# the truncated free-free basis converges, but slowly; clamped-end deflection
	# projects weakly onto free vibration modes. see README: static correction modes
	assert error[33] < error[12] < error[4]
