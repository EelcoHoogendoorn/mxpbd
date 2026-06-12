"""xpbd projection of rigid + modal degrees of freedom onto the constraint manifold

Two kinds of constraints share one lambda vector:
point constraints (the joints), organized into groups solved together as blocks,
and one elastic constraint per mode,
whose value is simply the modal amplitude, with compliance 1 / omega**2.
Treating the modal stiffness as xpbd compliance is what keeps arbitrarily stiff modes
unconditionally stable, with a well behaved rigid limit as omega -> inf.

The projection is least action: each substep seeks the smallest mass-weighted
displacement from the integrated prediction that satisfies the constraints,
each softened by its compliance:

	minimize  1/2 ds.T M ds + 1/2 sum C(s + ds)**2 / alpha

with alpha denoting each constraint's compliance over h**2 throughout
(the xpbd timestep scaling). Stationarity, with reactions lambda = -C / alpha:

	M ds = J.T lambda		the displacement is the reactions through the inverse mass
	C + alpha lambda = 0	each constraint holds, up to its compliance

Linearizing C about the prediction and eliminating ds = M_inv J.T lambda
leaves the dual system over the reactions alone:

	(J M_inv J.T + alpha) lambda = -C

Specialized to the dofs of this solver — per body a twist b and amplitudes q,
mass diag(Mb, I), and the modal jacobian the identity — this dual system is the
block system below, and everything past this point is bookkeeping its solution.

Each substep relaxes the totality of constraints gauss-seidel style:
first every modal constraint in closed form (a diagonal solve; for a body
without point constraints, this completes the substep), then each point group
as one coupled block over (point lambdas, modal lambdas):

	[[A, O], [O.T, B]] @ [dlp, dlm] = [-Cp - alpha_p lp, -Cm - alpha_m lm]

where:
	A = Jb Mb_inv Jb.T + Jq Mq_inv Jq.T + alpha_p	point-point coupling
	O = Jq Mq_inv									point-modal coupling (modal jacobian is identity)
	B = Mq_inv + alpha_m							diagonal

Solving the point lambdas of a group together with the modal lambdas is essential:
relaxing them in alternation converges poorly in the stiff limit.
Since B is diagonal, the modal lambdas are eliminated analytically:
the point constraints see each mode as a scalar mobility W = alpha_m / (1 + alpha_m),
interpolating between mass limited (soft modes) and immobile (the rigid limit),
and the schur complement over the group's point reactions reads

	S = Jb Mb_inv Jb.T + Jq W Jq.T + alpha_p

S is assembled over the constraint-body incidences only:
a point constraint touches at most two bodies,
and no zero jacobian blocks are formed or contracted anywhere.
The dense solve is over the group's point reactions only,
so a modal body costs the same as a rigid body in the solve,
regardless of its number of modes; and each group is localized to the bodies
its constraints reference, so a pass costs the sum of the group sizes.

Lambdas accumulate across the substep's passes (the incremental update of the
xpbd paper): the -alpha * lambda terms on the right hand side let overlapping
relaxations of the same constraint combine consistently - which is how the modal
rows, relaxed in the modal pass and again inside every group solve touching
their body, stay exactly converged rather than double-counted.
A single group holding all point constraints is the exact global solve;
per-joint groups trade coupling error per substep for smaller dense solves,
following standard pbd practice: substeps are the convergence knob, and each
group is relaxed once per substep.

A note on scale: everything here is python loops over lists of per-body pytrees,
which jax traces unrolled - per body in the integration and modal passes, per
group in the constraint pass. The graph thus grows with the scene; that is the right trade
for a demonstration codebase, where the code should mirror the math one to one,
but it becomes impractical somewhere in the hundreds of joints: compile time
first, device utilization second. The path to engine scale is known and would
not change the math: homogenize bodies and groups into stacked uniform classes
(padding away the ragged mode counts), and vmap the group solves over graph
colors, no two groups in a color sharing a body. The accumulated lambdas
already tolerate the jacobi flavor that batching introduces; convergence remains
a matter of substeps, as it is now. Deliberately not done here, in favor of
readability.
"""
import jax
from jax import numpy as jnp

from modal_xpbd.block import bop, solve_block_cholesky
from modal_xpbd.constraint import localize, modal_jacobian, residual, twist_jacobian


def jacobian_row(constraint, bodies):
	"""jacobian blocks of one point constraint, for the two bodies it touches

	constraint: PointConstraint, with body indices local to `bodies`
	bodies: [n_bodies] ModalBody; the bodies of this constraint's group

	Returns {b: (jb, jq)} over the touched bodies:
	jb: [2, 3], d(residual) / d(twist of body b)
	jq: [2, n_modes], d(residual) / d(amplitudes of body b)

	The two bodies of a constraint enter with opposite signs.
	The world body needs no special handling here: it has an ordinary jacobian,
	and zero inverse mass.
	"""
	c = constraint
	return {
		c.body_a: (
			twist_jacobian(bodies[c.body_a], c.anchor_a, c.modes_a, -1.0),
			modal_jacobian(bodies[c.body_a], c.modes_a, -1.0),
		),
		c.body_b: (
			twist_jacobian(bodies[c.body_b], c.anchor_b, c.modes_b, +1.0),
			modal_jacobian(bodies[c.body_b], c.modes_b, +1.0),
		),
	}


def modal_terms(bodies, previous, dt: float, damping: float):
	"""effective compliance and residual of the damped modal constraints

	Per mode, the dynamics to capture is the damped oscillator (unit modal mass),
	with `damping` the damping ratio zeta:

		q'' + 2 zeta omega q' + omega**2 q = f

	The spring is the xpbd constraint C = q with compliance 1 / omega**2
	(alpha, over h**2 as throughout); the dashpot is discretized implicitly,
	on the motion relative to `previous`, the bodies at the start of the substep.
	The reaction balancing spring plus dashpot at the end of the step satisfies

		q + g (q - q_prev) + alpha lambda = 0,		g = 2 zeta / (omega h)

	which divided through by 1 + g is again an undamped constraint row

		Cm + alpha_m lambda = 0

	with Cm = (q + g (q - q_prev)) / (1 + g) and alpha_m = alpha / (1 + g):
	the dashpot folds into the elastic constraint as residual and compliance
	scalings, preserving symmetry; and scaling both alike leaves static
	equilibria (q = q_prev) exactly untouched.
	Note that damping applied outside the solve is largely ineffective,
	since the projection recomputes the modal rates from realized positions.

	The modal compliance is the shape's; zero compliance (ReducedShape.rigid)
	degrades gracefully to an exactly rigid body.

	bodies, previous: [n_bodies] ModalBody, current and at the start of the substep

	Returns (alpha_m, Cm), both [n_bodies] of [n_modes]:
	effective compliance over dt**2, and effective constraint residual, per mode.
	Lists of [n_modes] arrays are ragged: each body brings its own mode count.
	"""
	gamma = [2 * damping * b.shape.omega * b.shape.compliance / dt for b in bodies]
	alpha_m = [
		b.shape.compliance / dt ** 2 / (1 + g)
		for b, g in zip(bodies, gamma)]
	Cm = [
		(b.amplitudes + g * (b.amplitudes - p.amplitudes)) / (1 + g)
		for b, p, g in zip(bodies, previous, gamma)]
	return alpha_m, Cm


def solve_constraints(bodies, constraints, dt, previous, lambdas,
		damping=0.0, regularization=1e-9):
	"""relax one non-empty group of point constraints,
	coupled with the modal constraints of the given bodies

	bodies: [n_bodies] ModalBody; the bodies referenced by this group's constraints,
		with constraint body indices local to this list (see localize)
	constraints: [n_constraints] PointConstraint
	previous: [n_bodies] ModalBody; the same bodies at the start of the substep
	lambdas: (lp, lm), the lambdas already accumulated this substep:
		lp: [n_constraints] of [2], point constraint lambdas
		lm: [n_bodies] of [n_modes], modal constraint lambdas
		incremental xpbd: these enter the right hand side as -alpha * lambda;
		with zero lambdas this is a plain single solve
	regularization: a small compliance added to every point constraint:
		it keeps the system regular when constraints are redundant,
		which constraint pairs on near-rigid bodies inherently are
		(a splice pair constrains the bolt-to-bolt distance, which only flex can absorb)

	Returns ((d_twist, d_amplitudes), (dlp, dlm)):
	d_twist: [n_bodies] of [3], position level twist displacements
	d_amplitudes: [n_bodies] of [n_modes], modal amplitude displacements
	dlp, dlm: lambda increments to accumulate, shaped as lp, lm
	"""
	lp, lm = lambdas

	Mb_inv = [b.twist_mass_inv() for b in bodies]			# [n_bodies] of [3]
	alpha_m, Cm = modal_terms(bodies, previous, dt, damping)	# [n_bodies] of [n_modes], twice
	alpha_p = [(c.compliance + regularization) / dt ** 2 for c in constraints]	# [n_constraints]
	# eliminating the diagonal modal block analytically (unit modal mass),
	# the point constraints see each mode as a scalar mobility W
	B_inv = [1 / (1 + am) for am in alpha_m]				# [n_bodies] of [n_modes]
	W = [am * bi for am, bi in zip(alpha_m, B_inv)]			# [n_bodies] of [n_modes]

	# body-major jacobian block matrices, [n_bodies][n_constraints];
	# each constraint contributes its own body slots. for a per-joint group
	# every constraint touches every group body and no zero blocks exist;
	# zeros pad only where a group mixes constraints over different body sets
	rows = [jacobian_row(c, bodies) for c in constraints]
	JbT = [
		[row[b][0].T if b in row else jnp.zeros((3, 2)) for row in rows]
		for b in range(len(bodies))]
	JqT = [
		[row[b][1].T if b in row else jnp.zeros((body.shape.n_modes, 2)) for row in rows]
		for b, body in enumerate(bodies)]

	Gp = [-(residual(c, bodies) + a * l) for c, a, l in zip(constraints, alpha_p, lp)]	# [n_constraints] of [2]
	Gm = [-(c + a * l) for c, a, l in zip(Cm, alpha_m, lm)]	# [n_bodies] of [n_modes]
	u = [bi * g for bi, g in zip(B_inv, Gm)]				# B_inv Gm; [n_bodies] of [n_modes]

	# schur complement over the point lambdas, and its right hand side
	S = bop('+',
		bop('ki,k,kj->ij', JbT, Mb_inv, JbT),
		bop('ki,k,kj->ij', JqT, W, JqT),
	)
	for i, a in enumerate(alpha_p):
		S[i][i] = S[i][i] + jnp.eye(2) * a
	rhs = bop('-', Gp, bop('ij,j->i', bop('ij->ji', JqT), u))
	dlp = solve_block_cholesky(S, rhs)						# [n_constraints] of [2]

	fb = bop('ij,j->i', JbT, dlp)							# Jb.T dlp; [n_bodies] of [3]
	fq = bop('ij,j->i', JqT, dlp)							# Jq.T dlp; [n_bodies] of [n_modes]
	dlm = bop('-', u, bop('*', B_inv, fq))					# [n_bodies] of [n_modes]
	d_twist = bop('*', Mb_inv, fb)							# [n_bodies] of [3]
	d_amplitudes = bop('+', fq, dlm)						# [n_bodies] of [n_modes]
	return (d_twist, d_amplitudes), (dlp, dlm)


def step(bodies, groups, dt, substeps: int = 1, gravity=(0.0, 0.0),
		damping=0.0, regularization=1e-9):
	"""advance the system by one timestep of dt, using a number of xpbd substeps

	groups is a list of non-empty point constraint groups; each group is relaxed
	as one coupled block, once per substep, gauss-seidel style across groups.
	All point constraints in a single group gives the exact global solve;
	smaller groups converge through substepping. A body claimed by no group
	is handled by the modal pass alone.

	Each substep integrates the unconstrained dofs forward,
	relaxes every modal constraint once in closed form, relaxes the point groups,
	and recovers velocities from the realized position change;
	that last velocity update is what makes substepping stable.
	"""
	g = jnp.asarray(gravity) * 1.0
	h = dt / substeps
	local_groups = localize(groups)

	def integrate(b):
		# gravity acts through the inverse mass like any force,
		# leaving the zero inverse mass world body unmoved
		dv = g * h * jnp.sign(b.shape.mass_inv)
		return b.replace(
			angle=b.angle + b.angular_velocity * h,
			position=b.position + (b.velocity + dv) * h,
			velocity=b.velocity + dv,
			amplitudes=b.amplitudes + b.rates * h,
		)

	def difference(b, p):
		# the inverse of integrate: velocities recovered from the realized positions
		return b.replace(
			angular_velocity=(b.angle - p.angle) / h,
			velocity=(b.position - p.position) / h,
			rates=(b.amplitudes - p.amplitudes) / h,
		)

	def substep(_, bodies):
		previous = bodies
		bodies = [integrate(b) for b in bodies]

		# modal pass: every modal constraint relaxed once, in closed form;
		# each mode is a constraint of its own, no block system involved.
		# this needs no place in the group pass below: the modal rows of touched
		# bodies are included in every group solve, and since they are exactly
		# linear in q, each solve leaves them exactly converged
		# (Cm + alpha_m * lm = 0); untouched bodies stay converged as left here
		alpha_m, Cm = modal_terms(bodies, previous, h, damping)
		lm = [-c / (1 + a) for c, a in zip(Cm, alpha_m)]
		bodies = [b.replace(amplitudes=b.amplitudes + l) for b, l in zip(bodies, lm)]

		# each group relaxed once: point lambdas start fresh, while the modal
		# lambdas carry over from the modal pass and any earlier group
		# touching the same body, keeping overlapping relaxations consistent
		for support, group in local_groups:
			(d_twist, d_amplitudes), (_, dlm) = solve_constraints(
				[bodies[j] for j in support], group, h,
				previous=[previous[j] for j in support],
				lambdas=([jnp.zeros(2) for _ in group], [lm[j] for j in support]),
				damping=damping, regularization=regularization)
			for k, j in enumerate(support):
				bodies[j] = bodies[j].displace(d_twist[k], d_amplitudes[k])
				lm[j] = lm[j] + dlm[k]

		return [difference(b, p) for b, p in zip(bodies, previous)]

	return jax.lax.fori_loop(0, substeps, substep, bodies)
