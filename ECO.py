import time

import numpy as np
import math


def levy(dim, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step


def close(t, G, X, G1Number, G2Number):
    m = X[0]
    if G == 1:
        for s in range(G1Number):
            if np.sum(np.abs(m - t)) > np.sum(np.abs(X[s] - t)):
                m = X[s]
    else:
        for s in range(G2Number):
            if np.sum(np.abs(m - t)) > np.sum(np.abs(X[s] - t)):
                m = X[s]
    return m


def ECO(Positions, fobj, lb, ub, Max_iter):
    N, dim = Positions.shape[0], Positions.shape[1]
    H = 0.5
    G1 = 0.2
    G2 = 0.1
    G1Number = int(N * G1)
    G2Number = int(N * G2)

    X = Positions
    fitness = np.array([fobj(X[i]) for i in range(N)])
    index = np.argsort(fitness)
    fitness = fitness[index]
    X = X[index]
    GBestF = fitness[0]
    AveF = np.mean(fitness)
    GBestX = X[0].copy()

    curve = np.zeros(Max_iter)
    avg_fitness_curve = np.zeros(Max_iter)
    search_history = np.zeros((N, Max_iter, dim))
    fitness_history = np.zeros((N, Max_iter))

    ct = time.time()
    for it in range(Max_iter):
        if it % 100 == 0 and it > 0:
            print(f"At iteration {it} the fitness is {curve[it - 1]}")
        avg_fitness_curve[it] = AveF
        R1, R2 = np.random.rand(), np.random.rand()
        P = 4 * np.random.randn() * (1 - it / Max_iter)
        E = (np.pi * it) / (P * Max_iter + 1e-10)
        w = 0.1 * np.log(2 - (it / Max_iter))
        X_new = X.copy()
        fitness_new = np.zeros(N)

        for j in range(N):
            if it % 3 == 0:  # Stage 1
                if j < G1Number:
                    X_new[j] = X[j] + w * (np.mean(X[j]) - X[j]) * levy(dim)
                else:
                    X_new[j] = X[j] + w * (close(X[j], 1, X, G1Number, G2Number) - X[j]) * np.random.randn()
            elif it % 3 == 1:  # Stage 2
                if j < G2Number:
                    X_new[j] = X[j] + (GBestX - np.mean(X, axis=0)) * np.exp(it / Max_iter - 1) * levy(dim)
                else:
                    if R1 < H:
                        tmp = close(X[j], 2, X, G1Number, G2Number)
                        X_new[j] = X[j] - w * tmp - P * (E * w * tmp - X[j])
                    else:
                        tmp = close(X[j], 2, X, G1Number, G2Number)
                        X_new[j] = X[j] - w * tmp - P * (w * tmp - X[j])
            else:  # Stage 3
                if j < G2Number:
                    X_new[j] = X[j] + (GBestX - X[j]) * np.random.randn() - (GBestX - X[j]) * np.random.randn()
                else:
                    if R2 < H:
                        X_new[j] = GBestX - P * (E * GBestX - X[j])
                    else:
                        X_new[j] = GBestX - P * (GBestX - X[j])

            X_new = np.clip(X_new[j], lb, ub)
            fitness_new[j] = fobj(X_new[j])
            if fitness_new[j] > fitness[j]:
                fitness_new[j] = fitness[j]
                X_new[j] = X[j].copy()
            if fitness_new[j] < GBestF:
                GBestF = fitness_new[j]
                GBestX = X_new[j].copy()

        X = X_new.copy()
        fitness = fitness_new.copy()
        curve[it] = GBestF
        search_history[:, it, :] = X
        fitness_history[:, it] = fitness
        index = np.argsort(fitness)
        X = X[index]
        fitness = fitness[index]

    Best_pos = GBestX
    Best_score = GBestF
    ct = time.time() - ct
    return Best_score, avg_fitness_curve, Best_pos, ct
