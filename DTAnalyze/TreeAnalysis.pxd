import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_float32 FLOAT_t
ctypedef np.npy_intp      INT_t

# Allow multiple datatypes
ctypedef fused DAT_TYPE_t:
    DOUBLE_t
    FLOAT_t
    INT_t

cpdef void DTLoadings(DAT_TYPE_t[:, :] A, INT_t m0, INT_t m1, DAT_TYPE_t[:, ::1] B,
                      INT_t[:] nLft, INT_t[:] nRit, INT_t[:] nFea, DAT_TYPE_t[:] nThr,
                      DAT_TYPE_t[:] nSam, DAT_TYPE_t[:] nVal) nogil

cpdef void FeatCounts(DAT_TYPE_t[:, :] A, INT_t m0, INT_t m1, DAT_TYPE_t[:, ::1] B,
                      INT_t[:] nLft, INT_t[:] nRit, INT_t[:] nFea, DAT_TYPE_t[:] nThr) nogil