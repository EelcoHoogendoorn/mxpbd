"""modal_xpbd: differentiable 2d rigid body dynamics enriched with modal flexibility

Self-contained illustration of XPBD-style block constraint projection
over bodies carrying both rigid (se2) and modal degrees of freedom.
Pure jax + numpy/scipy; no external geometry or physics dependencies.
"""
from modal_xpbd.truss import Truss, girder
from modal_xpbd.decompose import ReducedShape, reduce_modes
from modal_xpbd.body import ModalBody
from modal_xpbd.constraint import PointConstraint, pin, pin_world
from modal_xpbd.solve import solve_point_constraints, step
