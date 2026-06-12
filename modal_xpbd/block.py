"""Block matrix operations

These are all intended to be unrolled by jax, which influences the design;
no wrapping types, just plain nested lists passed into functions with specific input expectations.

Core invariant: for every operand, the nesting depth of the block structure equals the ndim of its leaves,
and both are dictated by the operand's term in the einsum formula.
That is, in 'ki,k,kj->ij' the first operand is a list-of-lists of 2d leaves,
the second a flat list of 1d leaves, and so on.
Every block level operation acts identically to flattening its operands
into non-blocked arrays and applying the same einsum formula there:

	flatten(bop(f, *blocks)) == jnp.einsum(f, *map(flatten, blocks))

This makes the formula string fully self-describing,
and gives a property test recipe for every formula.
"""
import numpy as np
from jax import numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


def bshape(block):
	if isinstance(block, (list, tuple)):
		return (len(block),) + bshape(block[0])
	return ()


def bdim(block):
	return len(bshape(block))


def bop(formula, *a):
	"""hardcoded block operations
	not very versatile but more self documented and testable this way

	each formula obeys the module-level invariant:
	the formula describes both the leaf einsum and the nesting structure of each operand,
	and the result equals applying the formula to the flattened operands
	"""
	assert all(bdim(q) > 0 for q in a)

	if formula == 'ki,k,kj->ij':
		# sandwich contraction with a diagonal middle term, as in J.T @ diag(m) @ J
		I, J, K = len(a[0][0]), len(a[2][0]), len(a[1])
		t = lambda i, j, k: jnp.einsum(formula, a[0][k][i], a[1][k], a[2][k][j])
		return [[sum(t(i, j, k) for k in range(K)) for j in range(J)] for i in range(I)]
	if formula == 'ij,jk->ik':
		# matmul
		I, J, K = len(a[0]), len(a[1]), len(a[1][0])
		t = lambda i, j, k: a[0][i][j] @ a[1][j][k]
		return [[sum(t(i, j, k) for j in range(J)) for k in range(K)] for i in range(I)]
	if formula == 'ij,j->i':
		# matvec
		I, J = len(a[0]), len(a[1])
		t = lambda i, j: a[0][i][j] @ a[1][j]
		return [sum(t(i, j) for j in range(J)) for i in range(I)]
	if formula == 'ij->ji':
		# transpose
		I, J = len(a[0]), len(a[0][0])
		return [[a[0][i][j].T for i in range(I)] for j in range(J)]
	if formula == 'ij,j->ij':
		# scale columns by a diagonal
		I, J = len(a[0]), len(a[1])
		return [[a[0][i][j] * a[1][j][None, :] for j in range(J)] for i in range(I)]

	binary = {
		'-': lambda x, y: x - y,
		'+': lambda x, y: x + y,
		'/': lambda x, y: x / y,
		'*': lambda x, y: x * y,
	}
	if formula in binary:
		op = binary[formula]
		if bdim(a[0]) == 1:
			I = len(a[0])
			return [op(a[0][i], a[1][i]) for i in range(I)]
		if bdim(a[0]) == 2:
			I, J = len(a[0]), len(a[0][0])
			return [[op(a[0][i][j], a[1][i][j]) for j in range(J)] for i in range(I)]
	raise NotImplementedError(formula)


def inner_shape(b):
	if bdim(b) == 1:
		return [q.shape[0] for q in b]
	if bdim(b) == 2:
		return [q[0].shape[0] for q in b], [q.shape[1] for q in b[0]]


def split_like(c, b):
	"""split flat array c into blocks shaped like block vector b"""
	s = np.cumsum(inner_shape(b))
	return jnp.split(c, s[:-1])


def solve_block_concat(M, v):
	"""reference dense solve; concatenate blocks, solve, split back up"""
	r = jnp.linalg.solve(jnp.block(M), jnp.block(v))
	return split_like(r, v)


def solve_block_cholesky(M, v):
	"""dense solve over concatenated blocks, by cholesky:
	the schur complements arising from the constraint saddle system
	are symmetric positive definite by construction"""
	r = cho_solve(cho_factor(jnp.block(M)), jnp.block(v))
	return split_like(r, v)


def solve_schur_diag(A, Binv, O, y, leaf_solver=solve_block_cholesky):
	"""Solve [[A, O], [O.T, B]] @ [xa, xb] = [ya, yb] for x, using schur complement of inv-diag B

	NOTE: it is assumed bdim(A)=2, bdim(Binv)=1, bdim(O)=2, bdim(ya)=bdim(yb)=1;
	Binv contains the inverted diagonals of the diagonal block B.
	The dense solve is over the size of A only;
	the diagonal B block is eliminated in linear time.
	"""
	ya, yb = y
	OBinv = bop('ij,j->ij', O, Binv)
	OT = bop('ij->ji', O)
	xa = leaf_solver(
		bop('-', A, bop('ij,jk->ik', OBinv, OT)),  # schur term
		bop('-', ya, bop('ij,j->i', OBinv, yb))
	)
	xb = bop('*', Binv, (bop('-', yb, bop('ij,j->i', OT, xa))))
	return [xa, xb]
