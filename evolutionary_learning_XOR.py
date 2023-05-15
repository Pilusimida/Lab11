import numpy as np
import random
import matplotlib.pyplot as plt

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# A function to perform a forward pass through the neural network. You can design your neural network however you want.
def predict(weights, inputs):
    layer1_weights = weights[:6].reshape(2, 3)
    layer2_weights = weights[6:].reshape(3, 1)

    layer1 = sigmoid(np.dot(inputs, layer1_weights))
    output = sigmoid(np.dot(layer1, layer2_weights))

    return output

# TODO: Implement a simple evolutionary learning algorithm to to optimize the weights of a feedforward neural network for the XOR problem.
# xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# xor_outputs = np.array([[0], [1], [1], [0]])
def fitness_function(weights, inputs, outputs):
    predictions = predict(weights, inputs)
    error = np.sum((predictions - outputs) ** 2)
    return 1 / (1 + error)

def mutate(weights, mutation_rate):
    for i in range(len(weights)):
        if random.random() < mutation_rate:
            weights[i] += np.random.normal(0, 0.5)
    return weights

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1))
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def genetic_algorithm(inputs, outputs, pop_size=50, num_generations=1000, mutation_rate=0.1):
    population = [np.random.randn(9) for _ in range(pop_size)]

    for generation in range(num_generations):
        fitness_values = [fitness_function(individual, inputs, outputs) for individual in population]
        top_individuals = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)[:pop_size//2]

        new_population = [individual for individual, fitness in top_individuals]
        while len(new_population) < pop_size:
            parent1 = random.choice(top_individuals)[0]
            parent2 = random.choice(top_individuals)[0]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    best_individual = max(population, key=lambda x: fitness_function(x, inputs, outputs))
    return best_individual

# XOR problem inputs and outputs
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([[0], [1], [1], [0]])

# Run the genetic algorithm to find the optimal weights for the XOR problem
optimal_weights = genetic_algorithm(xor_inputs, xor_outputs)
print("Optimal weights:", optimal_weights)

# Test the neural network with the optimal weights
predictions = predict(optimal_weights, xor_inputs)
print("Predictions:", predictions)
