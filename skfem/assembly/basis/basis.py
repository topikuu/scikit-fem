import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy import ndarray
from skfem.assembly.dofs import Dofs, DofsView
from skfem.element import DiscreteField, Element, ElementComposite
from skfem.mapping import Mapping
from skfem.mesh import Mesh
from skfem.quadrature import get_quadrature
from skfem.refdom import Refdom
from skfem.generic_utils import HashableNdArray

from functools import lru_cache

class Basis:
    """Finite element basis at global quadrature points.

    Please see the following implementations:

    - :class:`~skfem.assembly.InteriorBasis`, basis functions inside elements
    - :class:`~skfem.assembly.ExteriorFacetBasis`, basis functions on boundary
    - :class:`~skfem.assembly.InteriorFacetBasis`, basis functions on facets
      inside the domain

    """

    mesh: Mesh
    elem: Element
    tind: Optional[ndarray] = None
    # Integral weight per sample
    dx: ndarray
    # Basis evaluation samples at quadrature points.
    # List is over basis functions, the Tuple indexes over the element types
    # (in composite elements). The DiscreteField is a named tuple with names
    # value, grad, div, curl, hess, grad3, grad4, grad5 and grad6.
    # For example: self.basis[0][1].value are the values of first basis
    # function ([0]) of the second element type ([1]).
    # Note that the "first basis function" refers to the collection of basis
    # functions of all element types in a composite element. That is, only one
    # of the element types of the whole composite vector contains non-zero
    # values while all else are zeros.
    # All of these are [Type] x Ne x Nq ndarrays, where Ne is Number of elements
    # and Nq is number of quadrature points.
    # [Type] is the dimensionality of the considered field, e.g.
    # for .value it is 1 for simple elements, or 2 for a 2-vector-valued fields.
    # for .grad it is element's geometric dimension * .value's shape.
    basis: List[Tuple[DiscreteField, ...]] = []
    # Quadrature point positions in reference coordinates as a 2D array.
    # Each column is a single quadrature position.
    X: ndarray
    # Quadrature point weights as an 1D vector
    W: ndarray

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 refdom: Type[Refdom] = Refdom):

        if mesh.refdom != elem.refdom:
            raise ValueError("Incompatible Mesh and Element.")

        self.mapping = mesh._mapping() if mapping is None else mapping

        self.dofs = Dofs(mesh, elem)

        # global degree-of-freedom location
        try:
            doflocs = self.mapping.F(elem.doflocs.T)
            self.doflocs = np.zeros((doflocs.shape[0], self.N))

            # match mapped dofs and global dof numbering
            for itr in range(doflocs.shape[0]):
                for jtr in range(self.element_dofs.shape[0]):
                    self.doflocs[itr, self.element_dofs[jtr]] = doflocs[itr, :, jtr]
        except Exception:
            warnings.warn("Unable to calculate DOF locations.")

        self.mesh = mesh
        self.elem = elem

        self.Nbfun = self.element_dofs.shape[0]

        self.nelems = 0  # subclasses should overwrite

        if quadrature is not None:
            self.X, self.W = quadrature
        else:
            self.X, self.W = get_quadrature(
                refdom,
                intorder if intorder is not None else 2 * self.elem.maxdeg
            )

    @property
    def nodal_dofs(self):
        return self.dofs.nodal_dofs

    @property
    def facet_dofs(self):
        return self.dofs.facet_dofs

    @property
    def edge_dofs(self):
        return self.dofs.edge_dofs

    @property
    def interior_dofs(self):
        return self.dofs.interior_dofs

    @property
    def macro_dofs(self):
        return self.dofs.macro_dofs

    @property
    def N(self):
        return self.dofs.N

    @property
    def element_dofs(self):
        if self.tind is None:
            return self.get_element_dofs(None)
        else:
            return self.get_element_dofs(HashableNdArray(self.tind))

    @lru_cache(maxsize=128)
    def get_element_dofs(self, tind):
        if tind is None:
            return self.dofs.element_dofs
        return self.dofs.element_dofs[:, tind]

    def complement_dofs(self, *D):
        if type(D[0]) is dict:
            # if a dict of Dofs objects are given, flatten all
            D = tuple(D[0][key].all() for key in D[0])
        return np.setdiff1d(np.arange(self.N), np.concatenate(D))

    def find_dofs(self,
                  facets: Dict[str, ndarray] = None,
                  skip: List[str] = None) -> Dict[str, DofsView]:
        """Return global DOF numbers corresponding to a dictionary of facets.

        Facets can be queried from :class:`~skfem.mesh.Mesh` objects:

        >>> from skfem import MeshTri
        >>> m = MeshTri().refined()
        >>> m.facets_satisfying(lambda x: x[0] == 0)
        array([1, 5])

        This corresponds to a list of facet indices that can be passed over:

        >>> import numpy as np
        >>> from skfem import InteriorBasis, ElementTriP1
        >>> basis = InteriorBasis(m, ElementTriP1())
        >>> basis.find_dofs({'left': np.array([1, 5])})['left'].all()
        array([0, 2, 5])

        Parameters
        ----------
        facets
            A dictionary of facets. If ``None``, use ``self.mesh.boundaries``
            if set or otherwise use ``{'all': self.mesh.boundary_facets()}``.
        skip
            List of dofnames to skip.

        """
        if facets is None:
            if self.mesh.boundaries is None:
                facets = {'all': self.mesh.boundary_facets()}
            else:
                facets = self.mesh.boundaries

        return {k: self.dofs.get_facet_dofs(facets[k], skip_dofnames=skip)
                for k in facets}

    def get_dofs(self, facets: Optional[Any] = None,
                 skip_dofnames: List[str] = None) -> Any:
        """Find global DOF numbers.

        Accepts a richer set of types than
        :meth:`skfem.assembly.Basis.find_dofs`.

        Parameters
        ----------
        facets
            A list of facet indices. If ``None``, find facets by
            ``self.mesh.boundary_facets()``.  If callable, call
            ``self.mesh.facets_satisfying(facets)`` to get the facets.
            If array, simply find the corresponding DOF's. If a dictionary
            of arrays, find DOF's for each entry. If a dictionary of
            callables, call ``self.mesh.facets_satisfying`` for each entry to
            get facets and then find DOF's for those.

        """
        if facets is None:
            facets = self.mesh.boundary_facets()
        elif callable(facets):
            facets = self.mesh.facets_satisfying(facets)
        if isinstance(facets, dict):
            def to_indices(f):
                if callable(f):
                    return self.mesh.facets_satisfying(f)
                return f
            return {k: self.dofs.get_facet_dofs(to_indices(facets[k]),
                                                skip_dofnames)
                    for k in facets}
        return self.dofs.get_facet_dofs(facets, skip_dofnames)

    def default_parameters(self):
        """This is used by :func:`skfem.assembly.asm` to get the default
        parameters for 'w'."""
        raise NotImplementedError("Default parameters not implemented.")

    def interpolate(self, w: ndarray) -> Union[DiscreteField,
                                               Tuple[DiscreteField, ...]]:
        """Interpolate a solution vector to quadrature points.

        Useful when a solution vector is needed in the forms, e.g., when
        evaluating functionals or when solving nonlinear problems.

        Parameters
        ----------
        w
            A solution vector.

        """
        if w.shape[0] != self.N:
            raise ValueError("Input array has wrong size.")

        refs = self.basis[0]
        dfs: List[DiscreteField] = []

        # loop over solution components
        for c in range(len(refs)):
            ref = refs[c]
            fs = []

            def linear_combination(n, refn):
                """Global discrete function at quadrature points."""
                out = 0. * refn.copy()
                for i in range(self.Nbfun):
                    values = w[self.element_dofs[i]]
                    out += np.einsum('...,...j->...j', values,
                                     self.basis[i][c][n])
                return out

            # interpolate DiscreteField
            for n in range(len(ref)):
                if ref[n] is not None:
                    fs.append(linear_combination(n, ref[n]))
                else:
                    fs.append(None)

            dfs.append(DiscreteField(*fs))

        if len(dfs) > 1:
            return tuple(dfs)
        return dfs[0]

    def split_indices(self) -> List[ndarray]:
        """Return indices for the solution components."""
        if isinstance(self.elem, ElementComposite):
            o = np.zeros(5, dtype=np.int_)
            output: List[ndarray] = []
            for k in range(len(self.elem.elems)):
                e = self.elem.elems[k]
                output.append(np.concatenate((
                    self.nodal_dofs[o[0]:(o[0] + e.nodal_dofs)].flatten(),
                    self.edge_dofs[o[1]:(o[1] + e.edge_dofs)].flatten(),
                    self.facet_dofs[o[2]:(o[2] + e.facet_dofs)].flatten(),
                    self.interior_dofs[o[3]:(o[3] + e.interior_dofs)].flatten(),
                    self.macro_dofs[o[4]:(o[4] + e.macro_dofs)].flatten()
                )).astype(np.int_))
                o += np.array([e.nodal_dofs,
                               e.edge_dofs,
                               e.facet_dofs,
                               e.interior_dofs,
                               e.macro_dofs])
            return output
        raise ValueError("Basis.elem has only a single component!")

    def split_bases(self) -> List['Basis']:
        """Return Basis objects for the solution components."""
        if isinstance(self.elem, ElementComposite):
            return [type(self)(self.mesh, e, self.mapping,
                               quadrature=self.quadrature)
                    for e in self.elem.elems]
        raise ValueError("Basis.elem has only a single component!")

    @property
    def quadrature(self):
        return self.X, self.W

    def split(self, x: ndarray) -> List[Tuple[ndarray, 'Basis']]:
        """Split a solution vector into components."""
        xs = [x[ix] for ix in self.split_indices()]
        return list(zip(xs, self.split_bases()))

    def zero_w(self) -> ndarray:
        """Return a zero array with correct dimensions for
        :func:`~skfem.assembly.asm`."""
        return np.zeros((self.nelems, 0 if self.W is None else len(self.W)))

    def zeros(self) -> ndarray:
        """Return a zero array with same dimensions as the solution."""
        return np.zeros(self.N)

    def with_element(self, elem: Element) -> 'Basis':
        """Create a copy of ``self`` that uses different element."""
        raise NotImplementedError
