import numpy as np


# Define the objective function and constraints
def objective(x):
    x1, x2, x3, x4 = x
    return 10 * x1 ** 2 + 5 * x2 ** 2 + 10 * x3 ** 2 + x4 ** 2


def constraint1(x):
    x1, x2, x3, x4 = x
    return x1 + x2 + x3 + x4 - 40


def constraint2(x):
    x1, x2, x3, x4 = x
    return x1 ** 2 - x2 + x3 ** 2 - x4 + 5


def constraint3(x):
    x1, x2, x3, x4 = x
    return x1 + x3 - 20


def constraint4(x):
    x1, x2, x3, x4 = x
    return np.array([x1, x2, x3, x4])


# Define the boundaries
bounds = np.array([[-5, 5], [-5, 5], [-5, 5], [-5, 5]])


# Define the fitness function
def fitness(x):
    obj_value = objective(x)
    constraint_values = np.array([constraint1(x), constraint2(x), constraint3(x)])
    constraint_violations = np.maximum(0, -constraint_values)
    sum_violations = np.sum(constraint_violations)
    return obj_value + sum_violations


# Define the number of iterations and population size
n_iter = 100
pop_size = 10

# Initialize the population of solutions
pop = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(pop_size, 4))

# Initialize the best solution and fitness value
best_sol = pop[0]
best_fitness = fitness(best_sol)

# Initialize the velocities of the particles
velocities = np.zeros((pop_size, 4))

# Initialize the best positions and fitness values of the particles
best_positions = np.copy(pop)
best_fitnesses = np.array([fitness(sol) for sol in pop])

# Iterate for n_iter generations
for i in range(n_iter):
    # Update the velocities and positions of the particles
    for j in range(pop_size):
        # Update the velocity of the particle
        r1, r2 = np.random.uniform(size=2)
        velocities[j] = 0.5 * velocities[j] + 2 * r1 * (best_positions[j] - pop[j]) + 2 * r2 * (best_sol - pop[j])

        # Clamp the velocity to a maximum value
        max_velocity = 0.1 * (bounds[:, 1] - bounds[:, 0])
        velocities[j] = np.clip(velocities[j], -max_velocity, max_velocity)

        # Update the position of the particle
        pop[j] = pop[j] + velocities[j]

        # Clamp the position to the boundaries
        pop[j] = np.clip(pop[j], bounds[:, 0], bounds[:, 1])

        # Evaluate the fitness of the new position
        fitness_j = fitness(pop[j])

        # Update the best position and fitness of the particle if the new position is better
        if fitness_j < best_fitnesses[j]:
            best_positions[j] = pop[j]
            best_fitnesses[j] = fitness_j

        # Update the best position and fitness of the swarm if the new position is better
        if fitness_j < best_fitness:
            best_sol = pop[j]
            best_fitness = fitness_j

            # Print the best fitness value every 10 iterations
        if i % 10 == 0:
            print("Iteration {}: Best Fitness Value = {}".format(i, best_fitness))

        print("Final Best Fitness Value = {}".format(best_fitness))
        print("Final Best Solution = {}".format(best_sol))
