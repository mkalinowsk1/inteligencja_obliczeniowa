import numpy as np
import pygad
import time

item_name = ["zegar", "obraz-pejzaż", "obraz-portret", "radio", "laptop", "lampka nocna", "srebrne sztućce", "porcelana", "figura z brązu", "skórzana torebka", "odkurzacz"]
item_weight = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
item_value = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
knapsack_capacity = 25
best_fitness_value = 1630

def cal_fitness(ga_instance, solution, solution_idx):
	S1 = np.sum(solution * item_value)
	S2 = np.sum(solution * item_weight)
	if S2 <= knapsack_capacity:
		fitness = S1
	else:
		penalty = 100 * (S2 - knapsack_capacity)
		fitness = S1 - penalty
	return fitness

def pygad_optimization():
    
	num_generations = 200
	sol_per_pop = 50 
	num_parents_mating = int(sol_per_pop / 2) 

	start_time = time.time()
    
	ga_instance = pygad.GA(
		num_generations=num_generations,
		num_parents_mating=num_parents_mating,
		fitness_func=cal_fitness,
		sol_per_pop=sol_per_pop,
		num_genes=len(item_value),
		init_range_low=0,
		init_range_high=2,
		gene_type=int, 
		parent_selection_type="sss", 
		crossover_type="single_point", 
		mutation_type="random", 
		mutation_percent_genes=25, 
		stop_criteria=["reach_{}".format(best_fitness_value)] 
	)

	ga_instance.run()

	end_time = time.time()
	elapsed_time = end_time - start_time

	solution, solution_fitness, solution_idx = ga_instance.best_solution()

	# final_weight = np.sum(solution * item_weight)
	# final_value = solution_fitness

	# if final_weight > knapsack_capacity:
	# 	return 0

	# return final_value
     
	if solution_fitness == best_fitness_value:
		final_weight = np.sum(solution * item_weight)
		if final_weight <= knapsack_capacity:
			return elapsed_time
            
	return None

def pygad_time():
    successful_times = []
    
    print(f"Rozpoczynam {10} udanych prób znalezienia rozwiązania 1630 zł.")
    
    while len(successful_times) < 10:
        
        result_time = pygad_optimization()
        
        if result_time is not None:
            successful_times.append(result_time)
            print(f"Sukces ({len(successful_times)}/{10}): Czas: {result_time:.6f} s")
            
    average_time = np.mean(successful_times)
            
    print("\n--- WYNIK KOŃCOWY ---")
    print(f"Liczba udanych prób: {len(successful_times)}")
    print(f"Lista czasów z udanych prób: {successful_times}")
    print(f"Średni czas działania algorytmu do znalezienia rozwiązania 1630 zł: {average_time:.6f} s")
    
    return average_time

def main():
    correct_num = 0
    num_trials = 10
    
    print(f"Rozpoczynam {num_trials} przebiegów optymalizacji PyGAD...")
    
    for i in range(num_trials):
        result_value = pygad_optimization() 
        
        if result_value == best_fitness_value:
            correct_num += 1
            
    percentage = (correct_num / num_trials) * 100
    print("\n--- PODSUMOWANIE ---")
    print(f"Najlepsze rozwiązanie (1630) znaleziono {correct_num} na {num_trials} razy.")
    print(f"Procent dobrych odpowiedzi: {percentage:.2f}%")
    
    return percentage

if __name__ == "__main__":
    #main()
    pygad_time()