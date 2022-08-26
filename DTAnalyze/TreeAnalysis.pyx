# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
np.import_array()

# This function accumulates (in B) the changes in the node values (nVal) while traversing
# through a decision tree
cpdef void DTLoadings(DAT_TYPE_t[:, :] A, INT_t m0, INT_t m1, DAT_TYPE_t[:, ::1] B,
                      INT_t[:] nLft, INT_t[:] nRit, INT_t[:] nFea, DAT_TYPE_t[:] nThr,
                      DAT_TYPE_t[:] nSam, DAT_TYPE_t[:] nVal) nogil:
   cdef INT_t i, cn, j, ncn
   
   for i in range(m0, m1): # Loop over samples
       cn = 0
       while cn >= 0:      # Loop until leaf node
           j = nFea[cn]
           if j < 0:
              break
           # True is left; False is right
           ncn = nLft[cn] if (A[i, j] <= nThr[cn]) else nRit[cn]
           B[i, j] += (nVal[cn] - nVal[ncn])
           cn = ncn

cpdef void FeatCounts(DAT_TYPE_t[:, :] A, INT_t m0, INT_t m1, DAT_TYPE_t[:, ::1] B,
                      INT_t[:] nLft, INT_t[:] nRit, INT_t[:] nFea, DAT_TYPE_t[:] nThr) nogil:
   cdef INT_t i, cn, j, ncn
   
   for i in range(m0, m1): # Loop over samples
       cn = 0
       j  = nFea[0]
       while j >= 0:      # Loop until leaf node
           # True is left; False is right
           B[i, j] += 1
           cn = nLft[cn] if (A[i, j] <= nThr[cn]) else nRit[cn]
           if cn < 0:
              break