"""property tests of the block module invariant:
every block operation equals the same einsum applied to the flattened operands
"""
import jax
import numpy as np
from jax import numpy as jnp

from modal_xpbd.block import bop, solve_block_cholesky, solve_block_concat, solve_schur_diag

jax.config.update('jax_enable_x64', True)

rng = np.random.default_rng(0)


def blocks2(rows, cols):
	return [[jnp.asarray(rng.normal(size=(r, c))) for c in cols] for r in rows]


def blocks1(sizes):
	return [jnp.asarray(rng.normal(size=s)) for s in sizes]


def test_transpose():
	A = blocks2([2, 3], [4, 1, 2])
	assert jnp.allclose(jnp.block(bop('ij->ji', A)), jnp.block(A).T)


def test_matmul():
	A = blocks2([2, 3], [4, 1])
	B = blocks2([4, 1], [3, 2])
	assert jnp.allclose(jnp.block(bop('ij,jk->ik', A, B)), jnp.block(A) @ jnp.block(B))


def test_matvec():
	A = blocks2([2, 3], [4, 1])
	v = blocks1([4, 1])
	assert jnp.allclose(
		jnp.concatenate(bop('ij,j->i', A, v)),
		jnp.block(A) @ jnp.concatenate(v))


def test_diag_sandwich():
	J = blocks2([4, 1, 3], [2, 3])
	d = blocks1([4, 1, 3])
	assert jnp.allclose(
		jnp.block(bop('ki,k,kj->ij', J, d, J)),
		jnp.einsum('ki,k,kj->ij', jnp.block(J), jnp.concatenate(d), jnp.block(J)))


def test_column_scale():
	A = blocks2([2, 3], [4, 1])
	d = blocks1([4, 1])
	assert jnp.allclose(
		jnp.block(bop('ij,j->ij', A, d)),
		jnp.einsum('ij,j->ij', jnp.block(A), jnp.concatenate(d)))


def test_binary():
	import operator
	ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
	A = blocks2([2, 3], [4, 1])
	B = blocks2([2, 3], [4, 1])
	for f, op in ops.items():
		assert jnp.allclose(jnp.block(bop(f, A, B)), op(jnp.block(A), jnp.block(B)))


def test_solve_block_concat():
	n = [3, 2]
	A = blocks2(n, n)
	A = bop('+', A, bop('ij->ji', A))
	for i in range(len(n)):
		A[i][i] = A[i][i] + jnp.eye(n[i]) * 10
	y = blocks1(n)
	x = solve_block_concat(A, y)
	assert jnp.allclose(jnp.block(A) @ jnp.concatenate(x), jnp.concatenate(y))


def test_solve_block_cholesky():
	n = [3, 2]
	A = blocks2(n, n)
	A = bop('+', A, bop('ij->ji', A))
	for i in range(len(n)):
		A[i][i] = A[i][i] + jnp.eye(n[i]) * 10
	y = blocks1(n)
	assert jnp.allclose(
		jnp.concatenate(solve_block_cholesky(A, y)),
		jnp.concatenate(solve_block_concat(A, y)))


def test_solve_schur_diag():
	na, nb = [2, 2, 2], [5, 3]
	A = blocks2(na, na)
	A = bop('+', A, bop('ij->ji', A))
	# enough diagonal margin that the schur complement stays positive definite,
	# as the constraint systems served by the default cholesky leaf solver are
	for i in range(len(na)):
		A[i][i] = A[i][i] + jnp.eye(na[i]) * 30
	O = blocks2(na, nb)
	b = [jnp.abs(q) + 1 for q in blocks1(nb)]
	ya, yb = blocks1(na), blocks1(nb)

	xa, xb = solve_schur_diag(A, [1 / q for q in b], O, [ya, yb])

	H = jnp.block([
		[jnp.block(A), jnp.block(O)],
		[jnp.block(O).T, jnp.diag(jnp.concatenate(b))],
	])
	x = jnp.linalg.solve(H, jnp.concatenate([jnp.concatenate(ya), jnp.concatenate(yb)]))
	assert jnp.allclose(jnp.concatenate([jnp.concatenate(xa), jnp.concatenate(xb)]), x)
