import time

import numpy as np


# Snow Avalanches Algorithm (SAA)
def SAA(Positions, fitness_func, Xmin, Xmax, Itermax):
    NP, D = Positions.shape[0], Positions.shape[1]
    X = Positions
    si = 0.5
    fitness = np.array([fitness_func(ind) for ind in X])
    best_idx = np.argmin(fitness)
    XBest = X[best_idx].copy()
    fBest = fitness[best_idx]
    Convergence_curve = np.zeros((Itermax, 1))

    t = 0
    ct = time.time()
    while t < Itermax:
        for i in range(NP):
            indices = list(range(NP))
            indices.remove(i)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)
            Xr1, Xr2, Xr3 = X[r1], X[r2], X[r3]

            r = np.random.rand()
            if r < si:
                Xnew = XBest + np.random.rand(D) * (Xr1 - Xr2)
            elif r < 2 * si:
                Xnew = Xr3 + np.random.rand(D) * (Xr1 - Xr2)
            elif r < 3 * si:
                Xnew = X[i] + np.random.rand(D) * (Xr1 - Xr2)
            else:
                Xnew = X[i] + np.random.rand(D) * (Xmax - Xmin)
            Xnew = np.clip(Xnew, Xmin, Xmax)
            fnew = fitness_func(Xnew)
            if fnew[i] < fitness[i]:
                X[i] = Xnew[i]
                fitness[i] = fnew[i]

                # Update global best
                if fnew[i] < fBest:
                    XBest = Xnew[i]
                    fBest = fnew[i]
        Convergence_curve[t] = fBest
        t = t + 1
    fBest = Convergence_curve[Itermax - 1][0]
    ct = time.time() - ct
    return fBest, Convergence_curve, XBest, ct
