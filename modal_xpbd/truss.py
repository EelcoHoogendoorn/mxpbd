"""2d truss construction and assembly

Plain numpy; this is the offline preprocessing side of the pipeline.
Trusses are pin-jointed axial bars, which keeps the stiffness assembly trivial,
while bracing provides bending stiffness at the structure level.
"""
import dataclasses

import numpy as np


@dataclasses.dataclass
class Truss:
	vertices: np.ndarray	# [n, 2] node positions
	edges: np.ndarray		# [e, 2] int; axial bar elements
	stiffness: np.ndarray	# [e] axial stiffness EA of each bar
	density: float			# mass per unit length of each bar

	@property
	def n_nodes(self):
		return len(self.vertices)

	def deltas(self):
		return self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]]

	def lengths(self):
		return np.linalg.norm(self.deltas(), axis=1)

	def masses(self):
		"""[n] lumped nodal masses; half of each bar to each endpoint"""
		m = np.zeros(self.n_nodes)
		me = self.density * self.lengths() / 2
		np.add.at(m, self.edges[:, 0], me)
		np.add.at(m, self.edges[:, 1], me)
		return m

	def stiffness_matrix(self):
		"""[2n, 2n] assembled linear stiffness; vertex-major dof ordering"""
		L = self.lengths()
		d = self.deltas() / L[:, None]
		k = self.stiffness / L
		K = np.zeros((2 * self.n_nodes, 2 * self.n_nodes))
		for (i, j), ke, de in zip(self.edges, k, d):
			g = np.concatenate([-de, de])	# extension per unit dof motion
			idx = np.r_[2 * i:2 * i + 2, 2 * j:2 * j + 2]
			K[np.ix_(idx, idx)] += ke * np.outer(g, g)
		return K

	def gravity_force(self, gravity):
		"""[2n] nodal force vector under uniform acceleration"""
		return (self.masses()[:, None] * np.asarray(gravity)).flatten()

	def solve_static(self, fixed_nodes, force):
		"""[n, 2] static displacement under nodal force, with given nodes fully fixed.

		Full-order reference solution, for validating the reduced modal model against.
		"""
		K = self.stiffness_matrix()
		fixed = np.zeros(self.n_nodes, dtype=bool)
		fixed[np.asarray(fixed_nodes)] = True
		free = np.repeat(~fixed, 2)
		x = np.zeros(2 * self.n_nodes)
		x[free] = np.linalg.solve(K[np.ix_(free, free)], np.asarray(force)[free])
		return x.reshape(-1, 2)

	def solve_static_nonlinear(self, fixed_nodes, force, gtol=1e-9):
		"""[n, 2] static displacement with exact bar kinematics, by energy minimization.

		Geometrically exact full-order reference, for validating assembly level
		geometric nonlinearity, which the linear solve_static cannot capture.
		"""
		from scipy.optimize import minimize
		L0 = self.lengths()
		k = self.stiffness / L0
		f = np.asarray(force).reshape(-1, 2)
		fixed = np.zeros(self.n_nodes, dtype=bool)
		fixed[np.asarray(fixed_nodes)] = True
		free = ~fixed
		i, j = self.edges.T

		def unpack(u_free):
			u = np.zeros((self.n_nodes, 2))
			u[free] = u_free.reshape(-1, 2)
			return u

		def energy(u_free):
			d = np.diff((self.vertices + unpack(u_free))[self.edges], axis=1)[:, 0]
			L = np.linalg.norm(d, axis=1)
			return 0.5 * (k * (L - L0) ** 2).sum() - (f * unpack(u_free)).sum()

		def gradient(u_free):
			d = np.diff((self.vertices + unpack(u_free))[self.edges], axis=1)[:, 0]
			L = np.linalg.norm(d, axis=1)
			fe = (k * (L - L0) / L)[:, None] * d
			g = -f.copy()
			np.add.at(g, i, -fe)
			np.add.at(g, j, +fe)
			return g[free].flatten()

		result = minimize(
			energy, np.zeros(free.sum() * 2), jac=gradient, method='L-BFGS-B',
			options=dict(maxiter=50000, ftol=1e-16, gtol=gtol))
		residual = np.abs(result.jac).max() / max(np.abs(f).max(), 1e-30)
		assert residual < 1e-4, "static minimization did not converge"
		return unpack(result.x)


def girder(n_cells, width=1.0, height=1.0, stiffness=1.0, density=1.0) -> Truss:
	"""rectangular x-braced truss beam of n_cells unit cells"""
	nx = n_cells + 1
	x = np.arange(nx) * width
	vertices = np.concatenate([
		np.stack([x, np.zeros(nx)], axis=1),
		np.stack([x, np.full(nx, height)], axis=1),
	])
	bot, top = np.arange(nx), nx + np.arange(nx)
	edges = (
		[(bot[i], bot[i + 1]) for i in range(n_cells)] +
		[(top[i], top[i + 1]) for i in range(n_cells)] +
		[(bot[i], top[i]) for i in range(nx)] +
		[(bot[i], top[i + 1]) for i in range(n_cells)] +
		[(top[i], bot[i + 1]) for i in range(n_cells)]
	)
	edges = np.array(edges)
	return Truss(
		vertices=vertices,
		edges=edges,
		stiffness=np.full(len(edges), float(stiffness)),
		density=float(density),
	)
