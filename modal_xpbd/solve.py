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

from modal_xpbd.block import Block, beinsum, solve_block_cholesky
from modal_xpbd.constraint import localize, modal_jacobian, residual, twist_jacobian

# relative regularization floor for the point schur, as a fraction of the system
# scale; sets the worst-case condition number (~ 1 / RCOND) of the dense solve
RCOND = 1e-9


def twist_jacobians(constraints, bodies):
	"""body-major block matrix JbT of point-constraint twist jacobians, transposed

	Per constraint, d(residual) / d(twist) for its two bodies — body_a negated,
	body_b positive, as the residual is pb - pa — scattered into a
	[n_bodies][n_constraints] block matrix and padded with zeros where a
	constraint does not touch a body. For a per-joint group every constraint
	touches every group body and no padding arises; zeros appear only when a
	group mixes constraints over different body sets. The world body needs no
	special handling: an ordinary jacobian, with zero inverse mass.
	"""
	rows = [{
		c.body_a: -twist_jacobian(bodies[c.body_a], c.anchor_a, c.modes_a),
		c.body_b: twist_jacobian(bodies[c.body_b], c.anchor_b, c.modes_b),
	} for c in constraints]
	return Block([
		[row[b].T if b in row else jnp.zeros((3, 2)) for row in rows]
		for b in range(len(bodies))])


def modal_jacobians(constraints, bodies):
	"""body-major block matrix JqT of point-constraint modal jacobians, transposed

	d(residual) / d(amplitudes) per constraint, signed and scattered as in
	twist_jacobians; ragged per body in the mode count.
	"""
	rows = [{
		c.body_a: -modal_jacobian(bodies[c.body_a], c.modes_a),
		c.body_b: modal_jacobian(bodies[c.body_b], c.modes_b),
	} for c in constraints]
	return Block([
		[row[b].T if b in row else jnp.zeros((body.shape.n_modes, 2)) for row in rows]
		for b, body in enumerate(bodies)])


def modal_terms(bodies, previous, dt: float):
	"""effective compliance and residual of the damped modal constraints

	Per mode, the dynamics to capture is the damped oscillator (unit modal mass),
	with each body's `damping` the damping ratio zeta:

		q'' + 2 zeta omega q' + omega**2 q = f

	The spring is the xpbd constraint C = q with compliance 1 / omega**2
	(alpha, over h**2 as throughout); the dashpot is discretized implicitly,
	on the motion relative to `previous`, the bodies at the start of the substep.
	The reaction balancing spring plus dashpot at the end of the step satisfies

		q + g (q - q_prev) + alpha lambda = 0,		g = 2 zeta / (omega h)

	which divided through by 1 + g is again an undamped constraint row

		res_m + alpha_m lambda = 0

	with res_m = (q + g (q - q_prev)) / (1 + g) and alpha_m = alpha / (1 + g):
	the dashpot folds into the elastic constraint as residual and compliance
	scalings, preserving symmetry; and scaling both alike leaves static
	equilibria (q = q_prev) exactly untouched.
	Note that damping applied outside the solve is largely ineffective,
	since the projection recomputes the modal rates from realized positions.

	The modal compliance is the shape's; zero compliance (ReducedShape.rigid)
	degrades gracefully to an exactly rigid body.

	bodies, previous: [n_bodies] ModalBody, current and at the start of the substep

	Returns (alpha_m, res_m, B_inv) as Blocks, each [n_bodies] of [n_modes]:
	effective compliance over dt**2, effective constraint residual, and the inverse
	of the diagonal modal block B = modal mass inverse + alpha_m = 1 + alpha_m
	(unit modal mass plus compliance), the modal mobility shared by both the
	isolated modal solve and the coupled block solve.
	The blocks are ragged: each body brings its own mode count.
	"""
	gamma = [2 * b.damping * b.shape.omega * b.shape.compliance / dt for b in bodies]
	alpha_m = [
		b.shape.compliance / dt ** 2 / (1 + g)
		for b, g in zip(bodies, gamma)]
	res_m = [
		(b.amplitudes + g * (b.amplitudes - p.amplitudes)) / (1 + g)
		for b, p, g in zip(bodies, previous, gamma)]
	B_inv = [1 / (1 + am) for am in alpha_m]
	return Block(alpha_m), Block(res_m), Block(B_inv)


def solve_mode_constraint(bodies, previous, dt):
	"""relax every modal constraint in closed form, uncoupled from point constraints

	A body with no point constraint on it relaxes its modal constraints alone.
	Each mode is the constraint C = q (see modal_terms), its jacobian the identity
	I, with modal mass Mq = I. Solving jointly for the amplitude displacement and
	the reaction (d_amplitudes, lm) — stationarity Mq ds = I.T lm and the softened
	constraint I ds + alpha_m lm = -res_m — is the 2x2 system

		[ Mq   -I       ] [ d_amplitudes ]   [ .      ]
		[ I     alpha_m ] [ lm           ] = [ -res_m ]

	The top (stationarity) row gives the displacement from the reaction,
	d_amplitudes = Mq_inv lm; substituting into the bottom row collapses it to a
	scalar solve per mode,

		(Mq_inv + alpha_m) lm = -res_m   ->   lm = -res_m / (1 + alpha_m) = -res_m B_inv

	Mq is the identity, so Mq_inv = I and d_amplitudes = lm: they coincide
	numerically but are a (displacement, reaction) pair, not one quantity.

	Returns (d_amplitudes, lm), both [n_bodies] of [n_modes].
	"""
	_, res_m, B_inv = modal_terms(bodies, previous, dt)
	lm = -(res_m * B_inv)							# modal reaction
	d_amplitudes = lm								# Mq_inv lm; the modal mass is the identity
	return d_amplitudes.data, lm.data


def schur_solve(Jb, Jq, alpha_p, alpha_m, Gp, Gm, Mb_inv, B_inv):
	"""the coupled (point, modal) saddle solve, by reduction of the primal-dual system

	The step solves jointly for the displacement increments (d_twist, d_amplitudes)
	and the reaction increments (dlp, dlm): stationarity M ds = J.T lambda (the
	displacement is the reactions through the inverse mass) and the softened
	constraints J ds + alpha lambda = G. With the modal mass and the modal
	constraint jacobian both the identity I, that primal-dual system is

		[ Mb   .    -Jb.T    .       ] [ d_twist      ]   [ .  ]
		[ .    I    -Jq.T    -I      ] [ d_amplitudes ]   [ .  ]
		[ Jb   Jq    alpha_p  .      ] [ dlp          ] = [ Gp ]
		[ .    I     .        alpha_m] [ dlm          ]   [ Gm ]

	The top two (stationarity) rows give the displacements from the reactions,

		d_twist      = Mb_inv Jb.T dlp
		d_amplitudes = I      Jq.T dlp + dlm

	and substituting them into the bottom two leaves the dual saddle over the
	reactions alone (the modal block I + alpha_m is diagonal, inverse
	B_inv = 1 / (I + alpha_m)):

		[ Jb Mb_inv Jb.T + Jq Jq.T + alpha_p   Jq          ] [ dlp ]   [ Gp ]
		[ Jq.T                                 I + alpha_m ] [ dlm ] = [ Gm ]

	That block being diagonal, dlm is eliminated too,

		u   = B_inv Gm
		fq  = Jq.T dlp
		dlm = B_inv (Gm - fq) = u - B_inv fq

	leaving a dense schur complement over the point reactions alone,

		W   = alpha_m B_inv = alpha_m / (I + alpha_m)   per-mode mobility
		S   = Jb Mb_inv Jb.T + Jq W Jq.T + alpha_p
		dlp solves  S dlp = Gp - Jq u

	W runs from mass limited (soft modes) to immobile (the rigid limit, alpha_m ->
	0). The displacements then follow from the solved reactions,

		fb = Jb.T dlp,  d_twist = Mb_inv fb
		d_amplitudes = fq + dlm

	Jb, Jq are stored body-major (transposed), so the reactions Jb.T dlp, Jq.T dlp
	are coded Jb @ dlp, Jq @ dlp, and the rhs term Jq u is coded Jq.T @ u.
	"""
	W = B_inv * alpha_m										# modal mobility
	u = B_inv * Gm
	S = beinsum('ki,k,kj->ij', Jb, Mb_inv, Jb) + beinsum('ki,k,kj->ij', Jq, W, Jq)
	# relative regularization: a diagonal floor proportional to the system's own
	# scale, so the schur stays well conditioned in any mass/stiffness/dt regime
	# (an absolute floor only conditions one regime). The near-null direction of a
	# redundant splice pair is lifted to ~ RCOND * scale, bounding the condition
	# number near 1 / RCOND while leaving the rigid limit intact (joints stay rigid
	# to ~ RCOND). This is the smooth, spd-preserving cousin of a pinv rcond cutoff.
	scale = S.trace() / (2 * len(alpha_p))					# mean diagonal magnitude of the schur
	for i, a in enumerate(alpha_p):
		S[i][i] = S[i][i] + jnp.eye(2) * (a + RCOND * scale)	# physical compliance + relative floor
	dlp = solve_block_cholesky(S, Gp - Jq.T @ u)
	fb, fq = Jb @ dlp, Jq @ dlp
	dlm = u - B_inv * fq
	return (Mb_inv * fb, fq + dlm), (dlp, dlm)


def solve_point_constraints(bodies, constraints, dt, previous, lambdas):
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
	Each constraint's regularization (a small compliance floor, see PointConstraint)
	keeps the system regular when constraints are redundant, which constraint pairs
	on near-rigid bodies inherently are (a splice pair constrains the bolt-to-bolt
	distance, which only flex can absorb).

	Returns ((d_twist, d_amplitudes), (dlp, dlm)):
	d_twist: [n_bodies] of [3], position level twist displacements
	d_amplitudes: [n_bodies] of [n_modes], modal amplitude displacements
	dlp, dlm: lambda increments to accumulate, shaped as lp, lm
	"""
	# setup: physics to algebra
	point_lambda, modal_lambda = Block(lambdas[0]), Block(lambdas[1])
	twist_mass_inv = Block([b.twist_mass_inv() for b in bodies])
	modal_compliance, modal_residual, modal_block_inv = modal_terms(bodies, previous, dt)
	point_compliance = Block([c.compliance + c.regularization for c in constraints]) / dt ** 2
	point_residual = Block([residual(c, bodies) for c in constraints])
	twist_jac = twist_jacobians(constraints, bodies)
	modal_jac = modal_jacobians(constraints, bodies)
	# right hand sides: residuals plus the incremental -alpha * lambda terms
	point_rhs = -(point_residual + point_compliance * point_lambda)
	modal_rhs = -(modal_residual + modal_compliance * modal_lambda)

	(d_twist, d_amplitudes), (dlp, dlm) = schur_solve(
		Jb=twist_jac, Jq=modal_jac,
		alpha_p=point_compliance, alpha_m=modal_compliance,
		Gp=point_rhs, Gm=modal_rhs,
		Mb_inv=twist_mass_inv, B_inv=modal_block_inv,
	)
	return (d_twist.data, d_amplitudes.data), (dlp.data, dlm.data)


def step(bodies, groups, dt, substeps: int = 1, gravity=(0.0, 0.0)):
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

		# modal pass: relax every modal constraint once, in closed form. it needs
		# no place in the group pass below: the modal rows of touched bodies are
		# included in every group solve, and being exactly linear in q each solve
		# leaves them converged; untouched bodies stay converged as left here.
		d_amplitudes, lm = solve_mode_constraint(bodies, previous, h)
		bodies = [b.replace(amplitudes=b.amplitudes + da) for b, da in zip(bodies, d_amplitudes)]

		# each group relaxed once: point lambdas start fresh, while the modal
		# lambdas carry over from the modal pass and any earlier group
		# touching the same body, keeping overlapping relaxations consistent
		for support, group in local_groups:
			(d_twist, d_amplitudes), (_, dlm) = solve_point_constraints(
				[bodies[j] for j in support], group, h,
				previous=[previous[j] for j in support],
				lambdas=([jnp.zeros(2) for _ in group], [lm[j] for j in support]))
			for k, j in enumerate(support):
				bodies[j] = bodies[j].displace(d_twist[k], d_amplitudes[k])
				lm[j] = lm[j] + dlm[k]
				# since we only sweep once over each group delta-lambdas over point constraints are not accumulated

		return [difference(b, p) for b, p in zip(bodies, previous)]

	return jax.lax.fori_loop(0, substeps, substep, bodies)
