import warnings
from typing import Callable, Any, Optional
from functools import partial
from copy import deepcopy

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix

from ..basis import Basis
from ...element import DiscreteField


class FormDict(dict):
    """Passed to forms as 'w'."""
    # Immutable and hashable for caching of other functions as Python default
    # caching requires hashability of function arguments.
    def __init__(self, *args, **kwargs):
        super(FormDict, self).__init__(*args, **kwargs)
        self._hash = None


    def __getattr__(self, attr):
        return self[attr].value

    def __hash__(self):
        if self._hash is None:
            sorted_keys = tuple(sorted(self.keys()))
            self._hash = hash((sorted_keys, (self[k] for k in sorted_keys)))
        return self._hash

    def __setattr__(self, key, value):
        if key != '_hash':
            raise Exception('FormDict is immutable.')
        super(FormDict, self).__setattr__('_hash', value)

    def __setitem__(self, key, value):
        raise Exception('FormDict is immutable.')

class Form:

    form: Optional[Callable] = None

    def __init__(self,
                 form: Optional[Callable] = None,
                 dtype: type = np.float64):
        self.form = form.form if isinstance(form, Form) else form
        self.dtype = dtype

    def partial(self, *args, **kwargs):
        form = deepcopy(self)
        form.form = partial(form.form, *args, **kwargs)
        return form

    def __call__(self, *args):
        if self.form is None:  # decorate
            return type(self)(form=args[0], dtype=self.dtype)
        return self.assemble(self.kernel(*args))

    def assemble(self,
                 ubasis: Basis,
                 vbasis: Optional[Basis] = None,
                 **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def dictify(w):
        """Support additional input formats for 'w'."""
        for k in w:
            if isinstance(w[k], DiscreteField):
                continue
            elif isinstance(w[k], ndarray):
                w[k] = DiscreteField(w[k])
            elif isinstance(w[k], list):
                warnings.warn("In future, any additional kwargs to "
                              "asm() must be of type DiscreteField.",
                              DeprecationWarning)
                w[k] = DiscreteField(np.array([z.value for z in w[k]]),
                                     np.array([z.grad for z in w[k]]))
            elif isinstance(w[k], tuple):
                warnings.warn("In future, any additional kwargs to "
                              "asm() must be of type DiscreteField. "
                              "In most cases this deprecation is "
                              "fixed replacing asm(..., w=w) "
                              "by asm(..., w=DiscreteField(*w)).",
                              DeprecationWarning)
                w[k] = DiscreteField(*w[k])
            else:
                raise ValueError("The given type '{}' for the list of extra "
                                 "form parameters w cannot be converted to "
                                 "DiscreteField.".format(type(w)))
        return w

    @staticmethod
    def _assemble_scipy_matrix(data, rows, cols, shape=None):
        K = coo_matrix((data, (rows, cols)), shape=shape)
        K.eliminate_zeros()
        return K.tocsr()

    @staticmethod
    def _assemble_numpy_vector(data, rows, cols, shape=None):
        return coo_matrix((data, (rows, cols)),
                          shape=shape).toarray().T[0]
