import numpy as np
import random

# Define the graph
class Graph:
    def __init__(self, num_nodes, adjacency_matrix):
        self.num_nodes = num_nodes
        self.adjacency_matrix = adjacency_matrix

# Define the Genetic Algorithm for clustering
class GeneticAlgorithm:
    def __init__(self, graph, population_size, generations, mutation_rate):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        self.selection_probabilities = np.zeros(graph.num_nodes)
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = np.random.randint(0, 2, self.graph.num_nodes)
            population.append(chromosome)
        return population
    
    def fitness(self, chromosome):
        # Fitness function evaluating the quality of the clustering
        return np.sum(chromosome)  # Example: Number of cluster heads (this is a placeholder)
    
    def select_parents(self):
        fitness_values = np.array([self.fitness(chromosome) for chromosome in self.population])
        probabilities = fitness_values / fitness_values.sum()
        selected_indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]
    
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.graph.num_nodes - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    
    def mutate(self, chromosome):
        for i in range(self.graph.num_nodes):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome
    
    def run(self):
        for generation in range(self.generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population
            
            # Track the selection of each node as cluster head
            for chromosome in self.population:
                self.selection_probabilities += chromosome
        
        self.selection_probabilities /= (self.population_size * self.generations)
        return self.selection_probabilities

# Example usage
num_nodes = 10
adjacency_matrix = np.random.randint(0, 2, (num_nodes, num_nodes))
graph = Graph(num_nodes, adjacency_matrix)

ga = GeneticAlgorithm(graph, population_size=20, generations=100, mutation_rate=0.01)
selection_probabilities = ga.run()

print("Selection Probabilities of each node as cluster head:")
print(selection_probabilities)
