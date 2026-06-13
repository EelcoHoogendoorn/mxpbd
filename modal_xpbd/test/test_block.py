"""property tests of the block module invariant:
every block operation equals the same einsum applied to the flattened operands
"""
import jax
import numpy as np
from jax import numpy as jnp

from modal_xpbd.block import Block, beinsum, solve_block_cholesky, solve_block_concat, solve_schur_diag

jax.config.update('jax_enable_x64', True)

rng = np.random.default_rng(0)


def blocks2(rows, cols):
	return Block([[jnp.asarray(rng.normal(size=(r, c))) for c in cols] for r in rows])


def blocks1(sizes):
	return Block([jnp.asarray(rng.normal(size=s)) for s in sizes])


def test_transpose():
	A = blocks2([2, 3], [4, 1, 2])
	assert jnp.allclose(jnp.block(A.T.data), jnp.block(A.data).T)


def test_matmul():
	A = blocks2([2, 3], [4, 1])
	B = blocks2([4, 1], [3, 2])
	assert jnp.allclose(jnp.block((A @ B).data), jnp.block(A.data) @ jnp.block(B.data))


def test_matvec():
	A = blocks2([2, 3], [4, 1])
	v = blocks1([4, 1])
	assert jnp.allclose(
		jnp.concatenate((A @ v).data),
		jnp.block(A.data) @ jnp.concatenate(v.data))


def test_diag_sandwich():
	J = blocks2([4, 1, 3], [2, 3])
	d = blocks1([4, 1, 3])
	assert jnp.allclose(
		jnp.block(beinsum('ki,k,kj->ij', J, d, J).data),
		jnp.einsum('ki,k,kj->ij', jnp.block(J.data), jnp.concatenate(d.data), jnp.block(J.data)))


def test_column_scale():
	A = blocks2([2, 3], [4, 1])
	d = blocks1([4, 1])
	assert jnp.allclose(
		jnp.block(beinsum('ij,j->ij', A, d).data),
		jnp.einsum('ij,j->ij', jnp.block(A.data), jnp.concatenate(d.data)))


def test_pointwise():
	import operator
	A, B = blocks2([2, 3], [4, 1]), blocks2([2, 3], [4, 1])
	for block_op, op in [
		(Block.__add__, operator.add),
		(Block.__sub__, operator.sub),
		(Block.__mul__, operator.mul),
		(Block.__truediv__, operator.truediv),
	]:
		assert jnp.allclose(jnp.block(block_op(A, B).data), op(jnp.block(A.data), jnp.block(B.data)))
	assert jnp.allclose(jnp.block((-A).data), -jnp.block(A.data))


def test_solve_block_concat():
	n = [3, 2]
	A = blocks2(n, n)
	A = A + A.T
	for i in range(len(n)):
		A[i][i] = A[i][i] + jnp.eye(n[i]) * 10
	y = blocks1(n)
	x = solve_block_concat(A, y)
	assert jnp.allclose(jnp.block(A.data) @ jnp.concatenate(x.data), jnp.concatenate(y.data))


def test_solve_block_cholesky():
	n = [3, 2]
	A = blocks2(n, n)
	A = A + A.T
	for i in range(len(n)):
		A[i][i] = A[i][i] + jnp.eye(n[i]) * 10
	y = blocks1(n)
	assert jnp.allclose(
		jnp.concatenate(solve_block_cholesky(A, y).data),
		jnp.concatenate(solve_block_concat(A, y).data))


def test_solve_schur_diag():
	na, nb = [2, 2, 2], [5, 3]
	A = blocks2(na, na)
	A = A + A.T
	# enough diagonal margin that the schur complement stays positive definite,
	# as the constraint systems served by the default cholesky leaf solver are
	for i in range(len(na)):
		A[i][i] = A[i][i] + jnp.eye(na[i]) * 30
	O = blocks2(na, nb)
	b = [jnp.abs(q) + 1 for q in blocks1(nb)]
	ya, yb = blocks1(na), blocks1(nb)

	xa, xb = solve_schur_diag(A, Block([1 / q for q in b]), O, [ya, yb])

	H = jnp.block([
		[jnp.block(A.data), jnp.block(O.data)],
		[jnp.block(O.data).T, jnp.diag(jnp.concatenate(b))],
	])
	x = jnp.linalg.solve(H, jnp.concatenate([jnp.concatenate(ya.data), jnp.concatenate(yb.data)]))
	assert jnp.allclose(jnp.concatenate([jnp.concatenate(xa.data), jnp.concatenate(xb.data)]), x)
