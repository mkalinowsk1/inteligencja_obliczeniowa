import numpy as np
import random



def cal_fitness(weight, value, population, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight)
        if S2 <= threshold:
            fitness[i] = S1
        else :
            fitness[i] = 0 
    return fitness.astype(int)  

def selection(fitness, num_parents, population):
	fitness = list(fitness)
	parents = np.empty((num_parents, population.shape[1]))
	for i in range(num_parents):
		max_fitness_idx = np.where(fitness == np.max(fitness))
		parents[i,:] = population[max_fitness_idx[0][0], :]
		fitness[max_fitness_idx[0][0]] = -999999
	return parents

def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.8
    i=0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = random.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings    


def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = random.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = random.randint(0,offsprings.shape[1]-1)    
        if mutants[i,int_random_value] == 0 :
            mutants[i,int_random_value] = 1
        else :
            mutants[i,int_random_value] = 0
    return mutants   

def optimize(weight, value, population, pop_size, num_generations, threshold):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_generations):
        fitness = cal_fitness(weight, value, population, threshold)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
    print('Last generation: \n{}\n'.format(population)) 
    fitness_last_gen = cal_fitness(weight, value, population, threshold)      
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])
    return parameters, fitness_history

def main():
    item_name = ["zegar", "obraz-pejzaż", "obraz-portret", "radio", "laptop", "lampka nocna", "srebrne sztućce", "porcelana", "figura z brązu", "skórzana torebka", "odkurzacz"]
    item_weight = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
    item_value = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
    knapsack_capacity = 25

    ans = 0

    solutions_per_pop = 50
    pop_size = (solutions_per_pop, item_name.__len__())
    print('Population size = {}'.format(pop_size))
    initial_population = np.random.randint(2, size=pop_size)
    initial_population = initial_population.astype(int)
    num_generations = 200
    print('Initial population: \n{}'.format(initial_population))

    parameters, fitness_history = optimize(item_weight, item_value, initial_population, pop_size, num_generations, knapsack_capacity)
    parameters, fitness_history = optimize(item_weight, item_value, initial_population, pop_size, num_generations, knapsack_capacity)
    print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
    print('\nSelected items that will maximize the knapsack without breaking it:')
    for i in range(item_name.__len__()):
        if parameters[0][i] == 1:
            print(item_name[i])
            ans += item_value[i]
    return ans

if __name__ == "__main__":
    correct_num = 0
    for i in range(10):
            if main() == 1630:
                correct_num += 1
    print(f"procent dobrych odpowiedzi {correct_num/10 * 100}%")