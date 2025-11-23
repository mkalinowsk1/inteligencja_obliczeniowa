import numpy as np
import pygad
import time

MAZE = np.array([
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
	[1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
	[1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
	[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
	[1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
	[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
	[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
	[1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
	[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
	[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

])

NUM_GENES = 30
SOL_PER_POP = 200         
NUM_GENERATIONS = 5000     
NUM_PARENTS_MATING = 50  

GENE_SPACE = [0, 1, 2, 3]

MOVEMENT = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}

def fitness_func(ga_instance, solution, solution_idx):
	start_pos = tuple(np.argwhere(MAZE == 2)[0])
	end_pos = tuple(np.argwhere(MAZE == 3)[0])

	current_row, current_col = start_pos
	wall_hits = 0
	goal_reached = False

	for move in solution:
		dr, dc = MOVEMENT[int(move)]
		next_row, next_col = current_row + dr, current_col + dc

		if not(0 <= next_row < MAZE.shape[0] and 0 <= next_col < MAZE.shape[1]):
			wall_hits += 1
			continue
	
		cell_type = MAZE[next_row, next_col]

		if cell_type == 1: # Ściana
			wall_hits += 1
		elif cell_type == 3: # Cel
			current_row, current_col = next_row, next_col
			goal_reached = True
			break
		else:
			current_row, current_col = next_row, next_col

	if goal_reached:
		fitness = 1000.0 - (10 * wall_hits)
	else:
		manhattan_distance = abs(end_pos[0] - current_row) + abs(end_pos[1] - current_col)

		fitness = (100.0 - manhattan_distance) - (5.0 * wall_hits)
	return max(0.1, fitness)

def main():
	NUM_RUNS = 10
	execution_times = []
	found_solutions_count = 0

	for i in range(NUM_RUNS):
		ga_instance = pygad.GA(
			num_generations=NUM_GENERATIONS,
			num_parents_mating=NUM_PARENTS_MATING,
			fitness_func=fitness_func,
			sol_per_pop=SOL_PER_POP,
			num_genes=NUM_GENES,
			gene_space=GENE_SPACE,                 
			gene_type=int,                       
			parent_selection_type="sss",           
			crossover_type="single_point",         
			mutation_type='random',                
			mutation_percent_genes=0.05,
			stop_criteria=["reach_900.0"] 
		)

		start_time = time.time()

		ga_instance.run()

		end_time = time.time()
		elapsed_time = end_time - start_time

		best_solution, best_fitness, _ = ga_instance.best_solution()

		print(f"\n--- Uruchomienie {i+1} ---")
		print(f"Najlepsze fitness: {best_fitness:.2f}")
		print(f"Czas wykonania: {elapsed_time:.4f} s")


		if best_fitness > 900: 
			found_solutions_count += 1
			execution_times.append(elapsed_time)
			print("Rozwiązanie znalezione.")
		else:
			print("Rozwiązanie nie znalezione.")

		print("\n" + "="*50)
		print("WYNIKI KOŃCOWE")
		print("="*50)

		if execution_times:
			average_time = np.mean(execution_times)
			print(f"Liczba UDANYCH uruchomień (znaleziono drogę): {found_solutions_count} / {NUM_RUNS}")
			print(f"Średni czas znalezienia rozwiązania: {average_time:.4f} sekund.")
		else:
			print("Brak udanych uruchomień. Nie można obliczyć średniego czasu.")

if __name__ == '__main__':
	main()