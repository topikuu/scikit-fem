import unittest

import numpy as np

from skfem.assembly import InteriorBasis
from skfem.element import ElementTriP1
from skfem.mesh import MeshTri
from skfem.utils import projection


class InitializeScalarField(unittest.TestCase):
    def runTest(self):
        mesh = MeshTri().refined(5)
        basis = InteriorBasis(mesh, ElementTriP1())

        def fun(X):
            x, y = X
            return x ** 2 + y ** 2

        x = projection(fun, basis)
        y = fun(mesh.p)

        normest = np.linalg.norm(x - y)

        self.assertTrue(normest < 0.011,
                        msg="|x-y| = {}".format(normest))


if __name__ == '__main__':
    unittest.main()
