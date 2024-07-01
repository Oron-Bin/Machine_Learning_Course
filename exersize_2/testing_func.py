'''
First, we've generated random population, using the random function.
We've decided to use Binary Encoding, in order to decide which instrument will be used, and which won't. Each combination is a chromosome.
In order to calculate the fitness, we must make sure that the total weight of the given chromosome is smaller than the given W.
If an instrument in the given chromosome is 1, it adds the value of the instrument's weight, to a total weight parameter, and same for the value (total profit).
That way, we can assure that the weight isn't larger than W, and also sum the profit. If the weight exceeds the value of W, the fitness level is 0.  Then, mating selection- we select two chromosomes from a population for crossover. It calculates the fitness of each chromosome in the population (using the calculate fitness function.
Then, it normalizes the fitness values by dividing each value by the sum of all of the fitness values.  After that, it randomly select two chromosomes from the population, based on the normalized fitness values.  Then, two selected chromosomes are returned as parents.
Mate parents performs between two chromosomes.
It takes two chromosomes as input and randomly selects a point where it cuts the chromosome into two parts- 1a, 1b, 2a, 2b. By combining 2 parts, 2 children are born- 1a with 2b, and 2a with 1b.
Then two child chromosomes are returned as the output.  By choosing a random point, it extends the exploration process of the algorithm.
Mutation on the chromosome extends the algorithm exploitation.
It generates a random point between 0, and the length of the chromosome, and if the value of the point is 0, it changes it to 1, and the other way around.

The best chromosome is then being selected through a pool. It creates a list of fitness values, which represents the profit value.
The highest profit is what we're looking for, and for that reason, the iteration number is also being saved.
There's a control loop which evolves the population of the chromosomes. Two chromosomes are selected from the population, and then makes a crossover, is performed on them to generate two new chromosomes.
Then, the two new chromosomes are subjected to mutation with a given probability. The new chromosomes are individually subject to a random genetic mutation if the randomly generated probability is above the predetermined number.
Finally, the old population is replaced with the new population, which includes the two new chromosomes, and the remaining chromosomes from the old population.
'''

# import numpy as np
import random
# import matplotlib.pyplot as plt

#
instruments = [[1212, 2.91],
[1211, 8.19],
[612, 5.55],
[609, 15.6],
[1137, 3.70],
[1300, 13.5],
[585, 14.9],
[1225, 7.84],
[1303, 17.6],
[728, 17.3],
[211, 6.83],
[336, 14.4],
[894, 2.11],
[1381, 7.25],
[597, 4.65],
[858, 17.0],
[854, 7.28],
[1156, 5.01],
[597, 16.1],
[1129, 16.7],
[850, 3.10],
[874, 6.77],
[579, 10.7],
[1222, 1.25],
[896, 17.2]]
#
W = 200
n_population = 1000
mutation_probability = 0.3
# generations = 20


# function to generate a random population
def genesis(n_population):
    population = []
    for i in range(n_population):
        chromosome = []
        for j in range(len(instruments)):
            chromosome.append(random.randint(0,1))
        population.append(chromosome)
    return population

#to add valid chrmosome

# function to calculate the fitness of a chromosome
def fitness_eval(chromosome):
    total_profit = 0
    total_weight = 0
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            total_profit += instruments[i][0]
            total_weight += instruments[i][1]
    if total_weight > W:
        return 0
    else:
        return total_profit


# function to select two chromosomes for crossover
def mating_selection(population):
    fitness_values = []
    for chromosome in population:
        fitness_values.append(fitness_eval(chromosome))

    fitness_values = [float(i) / sum(fitness_values) for i in fitness_values]

    parent1 = random.choices(population, weights=fitness_values, k=1)[0]
    parent2 = random.choices(population, weights=fitness_values, k=1)[0]

    return parent1, parent2


# function that make crossover between 2 chromosome
def mate_parents(parent1, parent2):
    crossover_point = random.randint(0, len(instruments) - 1)
    child1 = parent1[0:crossover_point] + parent2[crossover_point:]
    child2 = parent2[0:crossover_point] + parent1[crossover_point:]

    return child1, child2


# function that make mutation due to random index that replaced
def mutate(chromosome):
    mutation_point = random.randint(0, len(instruments) - 1)
    if chromosome[mutation_point] == 0:
        chromosome[mutation_point] = 1
    else:
        chromosome[mutation_point] = 0
    return chromosome

def create_generation(population, mutation_rate):
    new_gen = []
    for i in range(0, len(population)):
        parent_1,parent_2 = mating_selection(population)
        child1,child2 = mate_parents(parent_1, parent_2)
        if random.random() < mutation_rate:
            child1 = mutate(child1)
            child2 = mutate(child2)

        new_gen.append(child1)
        new_gen.append(child2)
    return new_gen

# def best_solution(generation, instruments):
#     best = 0
#     for i in range(0, len(generation)):
#         temp = calculate_value(instruments, generation[i])
#         if temp > best:
#             best = temp
#     return best

# function to get the best chromosome from the population and the index of the best iteration
def get_best(population):
    fitness_values = []
    for chromosome in population:
        fitness_values.append(fitness_eval(chromosome))
    max_value = max(fitness_values)
    max_index= fitness_values.index(max_value)
    best_solution = [population[max_index], max_index + 1, fitness_values,max_value]
    return best_solution



#function that returns the solution in indexialy way

def convert_to_inst(list):
    instrument_list=[]
    for i in range(len(list)):
        if list[i] ==1:
            instrument_list.append(i)
    return instrument_list


#
# instruments = [[1212, 2.91],
# [1211, 8.19],
# [612, 5.55],
# [609, 15.6],
# [1137, 3.70],
# [1300, 13.5],
# [585, 14.9],
# [1225, 7.84],
# [1303, 17.6],
# [728, 17.3],
# [211, 6.83],
# [336, 14.4],
# [894, 2.11],
# [1381, 7.25],
# [597, 4.65],
# [858, 17.0],
# [854, 7.28],
# [1156, 5.01],
# [597, 16.1],
# [1129, 16.7],
# [850, 3.10],
# [874, 6.77],
# [579, 10.7],
# [1222, 1.25],
# [896, 17.2]]
#
# W = 200
# n_population = 1000
# mutation_probability = 0.3
# generations = 10

# generate a random population
population = genesis(n_population)


# create new populationts in terms of nember of generations
# history_max_fit=[]
# for k in range(generations):
#
#     parent1, parent2 = mating_selection(population)
#
#     child1, child2 = mate_parents(parent1, parent2)
#
#     # make mutation if a probablistic condintion is happen
#     if random.uniform(0, 1) < mutation_probability:
#         child1 = mutate(child1)
#     if random.uniform(0, 1) < mutation_probability:
#         child2 = mutate(child2)
#
#     population = [child1, child2] + population[2:] #update population
    # history_max_fit.append(get_best(population)[2])

best_sol = get_best(population)[0]
# print(history_max_fit)
# print(len(history_max_fit))
# print(get_best(population))
# get the weight and value of the best solution
total_profit = 0
total_weight = 0

for i in range(len(best_sol)):
    if best_sol[i] == 1:
        total_profit += instruments[i][0]
        total_weight += instruments[i][1]


#################### PLOT ###########################

# list_of_fitness = history_list
# print(len(list_of_fitness))
# # fitness_history_max = [np.max(fitness) for fitness in fitness_history]
# plt.scatter(list(range(generations)),history_list, label = 'Max Fitness')
# plt.legend()
# plt.title('Fitness through the generations')
# plt.xlabel('Generations')
# plt.ylabel('Fitness')
# plt.show()
# print(np.asarray(fitness_history).shape)
#################### SOLVE ###########################


# printing the result
print('GA found best solution after', get_best(population)[1] ,'iterations:')
print("Expected profit= $", total_profit)
print("Total weight= ", total_weight,'kg')
print('Included instruments are',convert_to_inst(get_best(population)[0]))
