"""Block matrices as nested lists of jax arrays

Intended to be unrolled by jax, which shapes the design: the leaves are plain
jax arrays and the block structure is plain nested lists, so it is all
transparent to tracing. `Block` is a thin convenience over such a nested list
(`.data` is the list itself); it is created and consumed within the traced
solve and never carried as loop state, so it needs no pytree registration.

Core invariant: for every block, nesting depth equals the ndim of its leaves.
A block vector is a flat list of leaves; a block matrix is a list of lists.
Pointwise arithmetic (+ - * /, unary -) acts leafwise and is spelled with the
operators; the .T property transposes a block matrix; multi-operand contractions
go through beinsum, whose einsum formula names both the leaf contraction and the
nesting. Every operation equals applying the same formula to the flattened
operands:

	flatten(beinsum(f, *bs)) == jnp.einsum(f, *map(flatten, bs))

which makes each formula self-describing and gives a property-test recipe.
"""
import operator

import numpy as np
from jax import numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


def _leafmap(f, d):
	return [_leafmap(f, x) for x in d] if isinstance(d, list) else f(d)


def _zip(op, x, y):
	if not isinstance(x, list):
		return op(x, y)
	if isinstance(y, list):
		return [_zip(op, a, b) for a, b in zip(x, y)]
	return [_zip(op, a, y) for a in x]		# broadcast a scalar against every leaf


class Block:
	"""A block vector or matrix: a nested list of jax arrays, depth == leaf ndim

	Pointwise arithmetic by operator, transpose by .T, contractions by beinsum.
	A thin convenience over the raw nested list, which is `.data`.
	"""
	__slots__ = ('data',)

	def __init__(self, data):
		self.data = data

	def __getitem__(self, i):
		return self.data[i]

	def __iter__(self):
		return iter(self.data)

	def __len__(self):
		return len(self.data)

	def __add__(self, o):
		return Block(_zip(operator.add, self.data, o.data if isinstance(o, Block) else o))

	def __sub__(self, o):
		return Block(_zip(operator.sub, self.data, o.data if isinstance(o, Block) else o))

	def __mul__(self, o):
		return Block(_zip(operator.mul, self.data, o.data if isinstance(o, Block) else o))

	def __truediv__(self, o):
		return Block(_zip(operator.truediv, self.data, o.data if isinstance(o, Block) else o))

	def __neg__(self):
		return Block(_leafmap(operator.neg, self.data))

	def __matmul__(self, o):
		# block matmul or matvec, by the right operand's depth
		return beinsum('ij,jk->ik' if isinstance(o.data[0], list) else 'ij,j->i', self, o)

	@property
	def T(self):
		d = self.data
		return Block([[d[i][j].T for i in range(len(d))] for j in range(len(d[0]))])


def beinsum(formula, *blocks):
	"""A multi-operand block contraction, named by its leaf einsum formula

	The formula describes both the per-leaf einsum and, by the module invariant,
	the nesting of each operand; the result equals the formula applied to the
	flattened operands. Pointwise ops and transpose live on Block; this is for
	the genuine contractions only.
	"""
	a = [b.data for b in blocks]
	if formula == 'ki,k,kj->ij':
		# sandwich contraction with a diagonal middle term, as in J.T @ diag(m) @ J
		I, J, K = len(a[0][0]), len(a[2][0]), len(a[1])
		t = lambda i, j, k: jnp.einsum(formula, a[0][k][i], a[1][k], a[2][k][j])
		return Block([[sum(t(i, j, k) for k in range(K)) for j in range(J)] for i in range(I)])
	if formula == 'ij,jk->ik':
		# matmul
		I, J, K = len(a[0]), len(a[1]), len(a[1][0])
		t = lambda i, j, k: a[0][i][j] @ a[1][j][k]
		return Block([[sum(t(i, j, k) for j in range(J)) for k in range(K)] for i in range(I)])
	if formula == 'ij,j->i':
		# matvec
		I, J = len(a[0]), len(a[1])
		t = lambda i, j: a[0][i][j] @ a[1][j]
		return Block([sum(t(i, j) for j in range(J)) for i in range(I)])
	if formula == 'ij,j->ij':
		# scale columns by a diagonal
		I, J = len(a[0]), len(a[1])
		return Block([[a[0][i][j] * a[1][j][None, :] for j in range(J)] for i in range(I)])
	raise NotImplementedError(formula)


def _split_like(c, template):
	"""split flat array c into a block vector shaped like the 1d block `template`"""
	s = np.cumsum([q.shape[0] for q in template])
	return Block(list(jnp.split(c, s[:-1])))


def solve_block_concat(M, v):
	"""reference dense solve; concatenate blocks, solve, split back up"""
	r = jnp.linalg.solve(jnp.block(M.data), jnp.block(v.data))
	return _split_like(r, v.data)


def solve_block_cholesky(M, v):
	"""dense solve over concatenated blocks, by cholesky:
	the schur complements arising from the constraint saddle system
	are symmetric positive definite by construction"""
	r = cho_solve(cho_factor(jnp.block(M.data)), jnp.block(v.data))
	return _split_like(r, v.data)


def solve_schur_diag(A, Binv, O, y, leaf_solver=solve_block_cholesky):
	"""Solve [[A, O], [O.T, B]] @ [xa, xb] = [ya, yb] via the schur complement of
	the inverse-diagonal B block

	Binv holds the inverted diagonals of the diagonal block B; the dense solve is
	over the size of A only, the diagonal B block eliminated in linear time.
	"""
	ya, yb = y
	OBinv = beinsum('ij,j->ij', O, Binv)
	OT = O.T
	xa = leaf_solver(
		A - OBinv @ OT,
		ya - OBinv @ yb)
	xb = Binv * (yb - OT @ xa)
	return [xa, xb]
