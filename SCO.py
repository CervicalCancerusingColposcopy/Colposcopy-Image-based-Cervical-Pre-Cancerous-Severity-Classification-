import numpy as np
import time

# Single candidate optimizer (SCO)
def perform_selection(population, fitness_values):
    sorted_indices = np.argsort(fitness_values)
    selected_population = population[sorted_indices]
    return selected_population


def perform_mutation(population, mutation_rate):
    mutated_population = population + mutation_rate * np.random.randn(*population.shape)
    return mutated_population


def perform_crossover(parents, crossover_rate):
    num_parents = len(parents)
    num_variables = parents.shape[1]
    offspring = np.zeros((num_parents, num_variables))
    for i in range(num_parents):
        for j in range(num_variables):
            if np.random.rand() < crossover_rate:
                offspring[i, j] = np.mean(parents[:, j])
            else:
                offspring[i, j] = parents[i, j]
    return offspring


def SCO(population, evaluate_fitness, lb, ub, max_iterations):
    mutation_rate = 0.1
    crossover_rate = 0.8
    pop_size, num_variables = population.shape
    best_solution = None
    best_fitness = float('inf')
    fitness_values = np.zeros(pop_size)
    ct = time.time()
    Convergence_curve = np.zeros((max_iterations, 1))

    for Iter in range(max_iterations):
        for i in range(pop_size):
            fitness_values[i] = evaluate_fitness(population[i])
        best_index = np.argmin(fitness_values)
        if fitness_values[best_index] < best_fitness:
            best_solution = population[best_index]
            best_fitness = fitness_values[best_index]

        selected_population = perform_selection(population, fitness_values)
        mutated_population = perform_mutation(selected_population, mutation_rate)
        offspring = perform_crossover(mutated_population, crossover_rate)

        population = offspring
        Convergence_curve[Iter] = best_fitness
    Time = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, Time
