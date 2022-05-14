from   joblib        import Parallel, parallel_backend, delayed
import numpy         as     np
from   .TreeAnalysis import GetDTLoadings

def GetFlatEstimators(M):
   if hasattr(M, 'tree_'):                # Looks like a tree
      return [M]
   if hasattr(M.estimators_, 'ravel'):    # Looks like GBM
      return M.estimators_.ravel()
   return M.estimators_                   # Looks like RF

def GetActivationsWrapper(EL, A, v, m0, m1, L):
   for Ei in EL:
      GetDTLoadings(A, 
                    m0,
                    m1,
                    L,
                    Ei.tree_.children_left,
                    Ei.tree_.children_right,
                    Ei.tree_.feature,
                    Ei.tree_.threshold,
                    Ei.tree_.weighted_n_node_samples,
                    getattr(Ei.tree_, v))
       
def GetActivations(M, A, v='impurity', m0=None, m1=None, n_jobs=1, lowMem=False):
   '''------------------------------------------------------------------------
   Estimates the contribution of each feature for determining the 
   ultimate predictions.
   M:      Decision tree, GBM, RF, or similar
   A:      The data matrix
   v:      The name of array field to use for calculating impurity reduction
   m0:     Starting index
   m1:     Ending index (exclusive)
   n_jobs: Number of threads (only helps on larger problems)
   lowMem: If False w/ n_jobs > 1, uses ~= n_jobs * A.shape extra memory
   -----------------------------------------------------------------------
   R:   A matrix (i, j) of importance of feature j in in sample i
   '''
   if m0 is None:
      m0 = 0
   if m1 is None:
      m1 = A.shape[0]
   EL = GetFlatEstimators(M)
   # Sum up feature activations for each tree 
   if n_jobs <= 1:
      L  = np.zeros(A.shape, order='C')
      GetActivationsWrapper(EL, A, v, m0, m1, L)
   else:
      if lowMem: # Each thread works on same output array
         # Each thread works on 1/n_jobs of array
         L  = np.zeros(A.shape, order='C')
         BL = SplitWork(A.shape[0], n_jobs)
         with parallel_backend('threading', n_jobs=n_jobs):
            # Process activations in parallel for each estimator
            Parallel()(delayed(GetActivationsWrapper)(EL, A, v, s, e, L) for s, e in BL)
      else: # Each thread works on its own output array
         n = len(EL)
         L = [np.zeros(A.shape, order='C') for _ in range(n_jobs)]
         # Each thread works on 1/n_jobs of estimators
         BL = SplitWork(n, n_jobs)
         with parallel_backend('threading', n_jobs=n_jobs):
            Parallel()(delayed(GetActivationsWrapper)(EL[s:e], A, v, m0, m1, L[i]) 
                                                          for i, (s, e) in enumerate(BL))
         L = sum(L)
               
   # NB that even w/ min_impurity_decrease=0.0 individual paths can increase impurity
   # since impurity decrease comes from both left and right split. Either individually
   # can increase impurity. This line makes decision to shift min to 0 and norm to 1
   L = L - np.minimum(0, L.min(1, keepdims=True))
   return L / L.sum(1, keepdims=True)

# Splits n tasks into k sequential blocks as evenly as possible
def SplitWork(n, k):
   r = n % k
   a = n // k   # (k - r) groups of size a
   b = 1 + a    # k       groups of size a+1==b
   e = r * b
   return ([(i * b, (i + 1) * b) for i in range(r)] + 
           [(e + i * a, e + (i + 1) * a) for i in range(k - r)])
