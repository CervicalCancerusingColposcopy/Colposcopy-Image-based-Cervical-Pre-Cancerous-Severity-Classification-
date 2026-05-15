import time
import numpy as np


# Fossa Optimization Algorithm (FOA)
def FOA(positions, obj_func, lb, ub, max_iter):
    pop_size, dim = positions.shape[0], positions.shape[1]

    # FOA parameters
    alpha = 2.0
    beta = 1.5
    gamma = 0.5
    p_ground = 0.3
    p_tree = 0.4

    fitness = obj_func(positions[:])
    territories = np.copy(positions)
    territory_strength = np.ones(pop_size)
    # Best solution
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_pos = positions[best_idx].copy()
    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    for iter in range(1, max_iter + 1):
        a = 2 - 2 * iter / max_iter  # Decreasing factor

        for i in range(pop_size):
            r = np.random.rand()

            if r < p_ground:
                # Ground hunting
                step_size = 0.1 * a * (ub - lb) * np.random.randn(dim)
                new_pos = positions[i] + step_size

            elif r < p_ground + p_tree:
                # Tree hunting
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                new_pos = (positions[i] +
                           alpha * r1 * (best_pos - positions[i]) +
                           beta * r2 * (territories[i] - positions[i]))
            else:
                # Ambush hunting
                if np.random.rand() < 0.5:
                    new_pos = positions[i]
                else:
                    target_idx = np.random.randint(pop_size)
                    while target_idx == i:
                        target_idx = np.random.randint(pop_size)
                    new_pos = (positions[i] +
                               gamma * np.random.rand(dim) *
                               (positions[target_idx] - positions[i]))

            # Territory influence
            if territory_strength[i] > np.random.rand():
                territory_influence = 0.2 * np.random.rand(dim) * (territories[i] - positions[i])
                new_pos += territory_influence

            # Social interaction
            if np.random.rand() < 0.1:
                neighbor_idx = np.random.randint(pop_size)
                while neighbor_idx == i:
                    neighbor_idx = np.random.randint(pop_size)
                social_influence = 0.1 * np.random.rand(dim) * (positions[neighbor_idx] - positions[i])
                new_pos += social_influence

            # Boundary handling
            new_pos = np.clip(new_pos, lb, ub)

            # Evaluate new position
            new_fitness = obj_func(new_pos)

            # Update if improved
            if new_fitness[i] < fitness[i]:
                positions[i] = new_pos[i]
                fitness[i] = new_fitness[i]
                territories[i] = new_pos[i]
                territory_strength[i] = min(territory_strength[i] + 0.1, 2)
            else:
                territory_strength[i] = max(territory_strength[i] - 0.05, 0.1)

            # Update global best
            if new_fitness[i] < best_fitness:
                best_fitness = new_fitness[i]
                best_pos = new_pos.copy()

        # Territory competition
        for i in range(pop_size):
            competitor_idx = np.random.randint(pop_size)
            while competitor_idx == i:
                competitor_idx = np.random.randint(pop_size)

            if fitness[i] < fitness[competitor_idx] and np.random.rand() < 0.3:
                territories[i] = 0.7 * territories[i] + 0.3 * territories[competitor_idx]

        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct
    return best_fitness, Convergence_curve, best_pos, ct
