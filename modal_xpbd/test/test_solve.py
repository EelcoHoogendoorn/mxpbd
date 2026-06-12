"""the structured solver against a naive monolithic reference

Any grouping or elimination strategy must agree with the one obviously-correct
formulation: assemble the full saddle system over the totality of
(point, modal) lambdas and solve it as a single dense linear system.
This pins down the schur elimination, the cholesky leaf solve,
the incremental -alpha * lambda terms, and the implicit damping fold,
all at machine precision.
"""
import jax
import numpy as np
from jax import numpy as jnp

from modal_xpbd.body import ModalBody
from modal_xpbd.constraint import pin, pin_world, residual
from modal_xpbd.decompose import reduce_modes
from modal_xpbd.solve import jacobian_row, modal_terms, solve_constraints
from modal_xpbd.truss import girder

jax.config.update('jax_enable_x64', True)

rng = np.random.default_rng(0)


def test_matches_monolithic_solve():
	shape = reduce_modes(girder(4, stiffness=200.0), 5)
	k = shape.n_modes
	bodies = [
		ModalBody.rest(shape, angle=a, position=((i + 0.5) * 4.0, 0.5))
		for i, a in enumerate([0.1, -0.2, 0.15])
	]
	bodies = [b.replace(amplitudes=jnp.asarray(rng.normal(size=k) * 0.1)) for b in bodies]
	# the world body enters the reference solve like any other body
	bodies = bodies + [ModalBody.world()]
	# distinct previous amplitudes, so the dashpot terms are nonzero
	previous = [b.replace(amplitudes=b.amplitudes * 0.8) for b in bodies]

	constraints = [
		pin(bodies, 0, 1, world_point=(4.0, 0.0)),
		pin(bodies, 0, 1, world_point=(4.0, 1.0)),
		pin(bodies, 1, 2, world_point=(8.0, 0.0)),
		pin_world(bodies, 0, world_point=(0.0, 0.0)),
	]
	dt, damping, regularization = 0.05, 0.3, 1e-9
	# nonzero accumulated lambdas, so the incremental terms are exercised
	lp = [jnp.asarray(rng.normal(size=2) * 0.01) for _ in constraints]
	lm = [jnp.asarray(rng.normal(size=b.shape.n_modes) * 0.01) for b in bodies]

	(d_twist, d_amplitudes), (dlp, dlm) = solve_constraints(
		bodies, constraints, dt, previous, (lp, lm),
		damping=damping, regularization=regularization)

	# the naive reference: one dense solve over the totality of lambdas,
	# densified from the sparse incidence rows
	n_b, n_c, n_m = len(bodies), len(constraints), sum(b.shape.n_modes for b in bodies)
	Jb = [[jnp.zeros((2, 3)) for _ in bodies] for _ in constraints]
	Jq = [[jnp.zeros((2, b.shape.n_modes)) for b in bodies] for _ in constraints]
	for i, row in enumerate([jacobian_row(c, bodies) for c in constraints]):
		for b, (jb, jq) in row.items():
			Jb[i][b] = jb
			Jq[i][b] = jq
	J = jnp.block([
		[jnp.block(Jb), jnp.block(Jq)],
		[jnp.zeros((n_m, 3 * n_b)), jnp.eye(n_m)],
	])
	alpha_m, Cm = modal_terms(bodies, previous, dt, damping)
	alpha = jnp.concatenate(
		[jnp.full(2, (c.compliance + regularization) / dt ** 2) for c in constraints] + alpha_m)
	m_inv = jnp.concatenate([b.twist_mass_inv() for b in bodies] + [jnp.ones(n_m)])
	H = jnp.einsum('id,d,jd->ij', J, m_inv, J) + jnp.diag(alpha)
	C = jnp.concatenate([residual(c, bodies) for c in constraints] + Cm)
	G = -(C + alpha * jnp.concatenate(lp + lm))
	dl = jnp.linalg.solve(H, G)
	du = m_inv * (J.T @ dl)

	assert jnp.allclose(jnp.concatenate(dlp + dlm), dl, atol=1e-9)
	assert jnp.allclose(jnp.concatenate(d_twist), du[:3 * n_b], atol=1e-9)
	assert jnp.allclose(jnp.concatenate(d_amplitudes), du[3 * n_b:], atol=1e-9)
