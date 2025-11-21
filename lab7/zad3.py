import numpy as np
import pygad
import math

def endurance(x, y, z, u, v, w):
    # math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)
    return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)

def fitness_func(ga_instance, solution, solution_idx):
    x, y, z, u, v, w = solution
    return endurance(x, y, z, u, v, w)

NUM_GENES = 6             
SOL_PER_POP = 100         
NUM_GENERATIONS = 100     
NUM_PARENTS_MATING = 50   


GENE_SPACE = {'low': 0.0, 'high': 1.0} 


MUTATION_PERCENT = 15 


ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=NUM_PARENTS_MATING,
    fitness_func=fitness_func,
    sol_per_pop=SOL_PER_POP,
    num_genes=NUM_GENES,
    gene_space=GENE_SPACE,                 
    gene_type=float,                       
    parent_selection_type="sss",           
    crossover_type="single_point",         
    mutation_type="random",                
    mutation_percent_genes=MUTATION_PERCENT 
)


print("Rozpoczynanie ewolucji...")
ga_instance.run()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
prediction = endurance(*solution)

print("\n--- WYNIKI OPTYMALIZACJI ---")
print(f"Najlepsze rozwiązanie (skład stopu): {solution}")
print(f"Najlepszy Fitness (teoretyczna wytrzymałość): {solution_fitness:.6f}")
print(f"Przewidywana wytrzymałość (z obliczeń): {prediction:.6f}")

ga_instance.plot_fitness()