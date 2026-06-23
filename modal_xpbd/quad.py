"""2d continuum meshes of bilinear (Q4) plane-stress elements

A drop-in alternative to truss.Truss for the modal pipeline: it exposes the same
interface reduce_modes consumes — vertices, edges, masses(), stiffness_matrix(),
n_nodes — so a quad mesh reduces to a ReducedShape with nothing else changed.

Where a Truss is pin-jointed bars (a lattice that only approximates a continuum,
and whose bending compliance can leak into the joints), a QuadMesh is a genuine
solid: bending lives in the material, governed by a modulus E and poisson ratio
nu. That makes it a faithful model of a real steel section, in real units.

Plain numpy; the offline preprocessing side of the pipeline. Full 2x2 gauss
integration (no reduced integration), so there are no hourglass modes: a free
mesh has exactly the three rigid-body modes and no spurious mechanisms.
"""
import dataclasses

import numpy as np

# 2-point gauss nodes on [-1, 1]; the weights are both 1
_GAUSS = (-1 / np.sqrt(3), 1 / np.sqrt(3))
# element corners, counterclockwise from bottom-left, in local (xi, eta)
_CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])


def _element_stiffness(a, b, E, nu):
	"""[8, 8] plane-stress stiffness of an a-by-b rectangular Q4 element (unit thickness)

	dofs ordered [u0, v0, u1, v1, u2, v2, u3, v3] over the four corners,
	counterclockwise from bottom-left; integrated with 2x2 gauss.
	"""
	D = E / (1 - nu ** 2) * np.array([
		[1, nu, 0],
		[nu, 1, 0],
		[0, 0, (1 - nu) / 2],
	])
	Ke = np.zeros((8, 8))
	for xi in _GAUSS:
		for eta in _GAUSS:
			# shape function derivatives wrt physical x, y at this gauss point
			dN_dx = 0.25 * _CORNERS[:, 0] * (1 + _CORNERS[:, 1] * eta) * (2 / a)
			dN_dy = 0.25 * _CORNERS[:, 1] * (1 + _CORNERS[:, 0] * xi) * (2 / b)
			B = np.zeros((3, 8))
			B[0, 0::2] = dN_dx			# exx = du/dx
			B[1, 1::2] = dN_dy			# eyy = dv/dy
			B[2, 0::2] = dN_dy			# gxy = du/dy + dv/dx
			B[2, 1::2] = dN_dx
			Ke += (B.T @ D @ B) * (a / 2) * (b / 2)		# detJ = (a/2)(b/2), gauss weights 1
	return Ke


@dataclasses.dataclass
class QuadMesh:
	vertices: np.ndarray	# [n, 2] node positions
	elements: np.ndarray	# [e, 4] int; corner nodes, counterclockwise
	edges: np.ndarray		# [b, 2] int; element boundary segments, for rendering
	a: float				# element width
	b: float				# element height
	E: float				# young's modulus
	nu: float				# poisson ratio
	density: float			# mass per unit area (mass per volume at unit thickness)

	@property
	def n_nodes(self):
		return len(self.vertices)

	def masses(self):
		"""[n] lumped nodal masses; each element's mass split equally to its corners"""
		m = np.zeros(self.n_nodes)
		np.add.at(m, self.elements.ravel(), self.density * self.a * self.b / 4)
		return m

	def stiffness_matrix(self):
		"""[2n, 2n] assembled plane-stress stiffness; vertex-major dof ordering"""
		Ke = _element_stiffness(self.a, self.b, self.E, self.nu)
		K = np.zeros((2 * self.n_nodes, 2 * self.n_nodes))
		for elem in self.elements:
			dofs = np.repeat(elem * 2, 2) + np.tile([0, 1], 4)	# [2n0, 2n0+1, 2n1, ...]
			K[np.ix_(dofs, dofs)] += Ke
		return K

	def gravity_force(self, gravity):
		"""[2n] nodal force vector under uniform acceleration"""
		return (self.masses()[:, None] * np.asarray(gravity)).flatten()


def quad_strip(nx, ny=1, cell=1.0, E=1.0, nu=0.3, density=1.0) -> QuadMesh:
	"""a rectangular plate of nx by ny square cells of side `cell`

	A thin strip (ny small, nx large) is a 2d beam / steel wire: pass steel's
	E, nu, density and it carries genuine continuum bending in real units.
	"""
	nvx, nvy = nx + 1, ny + 1
	grid = np.stack(np.meshgrid(np.arange(nvx), np.arange(nvy), indexing='ij'), axis=-1)
	vertices = grid.reshape(-1, 2).astype(float) * cell

	def node(i, j):
		return i * nvy + j

	elements = np.array([
		[node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
		for i in range(nx) for j in range(ny)
	])
	segments = {
		tuple(sorted((int(e[k]), int(e[(k + 1) % 4]))))
		for e in elements for k in range(4)
	}
	edges = np.array(sorted(segments))

	return QuadMesh(
		vertices=vertices, elements=elements, edges=edges,
		a=float(cell), b=float(cell), E=float(E), nu=float(nu), density=float(density),
	)
