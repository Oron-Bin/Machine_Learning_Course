import numpy as np
import random
import matplotlib.pyplot as plt

n_population = 100
mutation_rate = 0.4
alpha = 0.5 # Parameter indicating where to split the parent in the crossover - can also be random

# First step: Create the first population set
def genesis(city_names, n_population):

    population_set = []
    for i in range(n_population):
        #Randomly generating a new solution
        sol_i = ''.join(random.sample(city_names, len(city_names)))
        population_set.append(sol_i)
    return np.array(population_set)

#################### Fitness ###########################

def distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

# Fitness of a path
def fitness_eval(path, cities_list):
    n_cities = len(cities_list)
    total = 0
    for i in range(1, n_cities):
        total += distance(cities_list[path[i-1]], cities_list[path[i]])
    return total

def get_all_fitness(population, cities_dict):
    n_population = len(population)
    fitness_list = np.zeros(n_population)

    #Looping over all solutions computing the fitness for each solution
    for i in  range(n_population):
        fitness_list[i] = fitness_eval(population[i], cities_dict)

    return fitness_list

#################### SELECTION ###########################

def Mating_selection(population, fitness_list):
    # Compute probability list for selecting parents - large fitness provides higher probability
    total_fit = fitness_list.sum()
    prob_list = 1 - (fitness_list/total_fit) 
    prob_list = prob_list/prob_list.sum()
    
    # Notice there is the chance that a parent mates with oneself
    parent_list_a = np.random.choice(list(range(len(population))), len(population),p=prob_list, replace=True)
    parent_list_b = np.random.choice(list(range(len(population))), len(population),p=prob_list, replace=True)
    
    parent_list_a = population[parent_list_a]
    parent_list_b = population[parent_list_b]
    
    return np.array([parent_list_a,parent_list_b])

#################### CROSSOVER ###########################

# We take part (head_a) of parent_a and mate it with another part (tail) in parent_b. However, some repetitions may occur. Hence, we replace cities in head_a with cities in tail_a that are not in tail_b
# Refer to the pairing in: https://towardsdatascience.com/an-extensible-evolutionary-algorithm-example-in-python-7372c56a557b
def mate_parents(parent_a, parent_b):

    # alpha = np.random.random() * 0.7 + 0.2 # Can be constant or randomized

    head_a = parent_a[:int(len(parent_a)*alpha)]
    tail_a = parent_a[int(len(parent_a)*alpha):]
    tail_b = parent_b[int(len(parent_b)*alpha):]

    mapping = {tail_b[i]: tail_a[i] for i in range(len(tail_a))}

    for i in range(len(head_a)):
        while head_a[i] in tail_b:
            head_a = head_a.replace(head_a[i], mapping[head_a[i]] )

    return head_a + tail_b
            
    
def mate_parents_list(Mating_list):
    new_population = []
    for i in range(Mating_list.shape[1]):
        parent_a, parent_b = Mating_list[0][i], Mating_list[1][i]
        offspring = mate_parents(parent_a, parent_b)
        new_population.append(offspring)
        
    return new_population

#################### MUTATE #########################

# For each element of the new population, we add a random chance of swapping
def mutate_offspring(offspring, n_cities):
    offspring = list(offspring)
    for q in range(int(n_cities*mutation_rate)):
        a = np.random.randint(0, n_cities)
        b = np.random.randint(0, n_cities)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return ''.join(offspring)
    
    
def mutate_population(new_population, n_cities):
    mutated_pop = []
    for offspring in new_population:
        mutated_pop.append(mutate_offspring(offspring, n_cities))
    return np.array(mutated_pop)

#################### PLOT ###########################

def plot_path(cities_list, path, fitness):

    fig = plt.figure(1, figsize=(6, 4))
    fig.clf()

    loc = np.array(list(cities_list.values()))
    plt.scatter(x=loc[:, 0], y=loc[:, 1], s=500, zorder=1)

    for city in cities_list.keys():
        plt.text(cities_list[city][0], cities_list[city][1], city, horizontalalignment='center', verticalalignment='center', size=10, c='white')

    for i in range(len(path)-1):
        plt.plot([cities_list[path[i]][0], cities_list[path[(i + 1)]][0]],
                 [cities_list[path[i]][1], cities_list[path[(i + 1)]][1]], 'k', zorder=0)
    plt.title(f'Visiting {len(path)} cities in distance {fitness:.2f}', size=16)
    # plt.show()

#################### SOLVE ###########################

def solve(cities_list):

    n_cities = len(cities_list)
    population = genesis(list(cities_list.keys()), n_population)

    best_solution = [-1, np.inf, []]
    BEST = []
    
    for i in range(1, 5000):
        if i % 100 == 0: 
            print(i, fitness_list.min(), fitness_list.mean(), best_solution[1])

            # fig = plt.figure(0)
            # fig.clf()
            # plt.plot(BEST, 'k')
            # plot_path(cities_list, best_solution[2], best_solution[1])
            # plt.pause(0.0001)

        fitness_list = get_all_fitness(population, cities_list)
        
        #Saving the best solution
        if fitness_list.min() < best_solution[1]:
            best_solution[0] = i
            best_solution[1] = fitness_list.min()
            best_solution[2] = population[fitness_list.min() == fitness_list][0]

        Mating_list = Mating_selection(population, fitness_list)
        new_population = mate_parents_list(Mating_list)
        
        population = mutate_population(new_population, n_cities)
        
        BEST.append(best_solution[1])

        if i > 3000 and np.all(np.array(BEST[:-2000]) == BEST[-1]):
            break

    return best_solution

#####################################################

cities_list = {'A': [35, 51],
                   'B': [113, 213],
                   'C': [82, 280],
                   'D': [322, 340],
                   'E': [256, 352],
                   'F': [160, 24],
                   'G': [322, 145],
                   'H': [12, 349],
                   'I': [282, 20],
                   'J': [241, 8],
                   'K': [398, 153],
                   'L': [182, 305],
                   'M': [153, 257],
                   'N': [275, 190],
                   'O': [242, 75],
                   'P': [19, 229],
                   'Q': [303, 352],
                   'R': [39, 309],
                   'S': [383, 79],
                   'T': [226, 343]}

sol = solve(cities_list)

print("Best solution:", sol)
plot_path(cities_list, sol[2], sol[1])
plt.show()



