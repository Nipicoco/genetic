#       in loving memory of 
#
#       population = [population[j] + Direcciones[direction[j]] if((1 <= population[j][0] + Direcciones[direction[j]][0] <=tam and 1 <= population[j][1] + Direcciones[direction[j]][1] <=tam) and not((population[j][0]) == tam) and comprobacion(np.add(population[j], Direcciones[direction[j]]), population)) else population[j] + Direcciones[4] for j in range(len(population))]
#       </3 :(
#💥🍑👋💥
import numpy as np
import matplotlib.pyplot as plt
import random

Direcciones = [
 [-1,1], # NOROESTE, Blue        
 [0,1],  # NORTE, Green
 [1,1],  # NORESTE, Red
 [-1,0], # OESTE, Cyan
 [0,0],  # NO MUEVE ESE POTO, Magenta
 [1,0],  # ESTE, Yellow
 [-1,-1],# SUROESTE, Black
 [0,-1], # SUR, Purple
 [1,-1]  # SURESTE, Orange
]
Direcciones = np.array(Direcciones)
Colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange']

def Color(cromosoma): #[0.8, 0.15, 0.009, 0.3, 0.1]
    max = np.argmax(cromosoma)
    return Colors[max]

def Cromosoma():
    return np.random.uniform(0,1,9)

def CromosomaNormalizada(cromosoma):
    Suma = np.sum(cromosoma)
    for i in range(len(cromosoma)):
        cromosoma[i] = cromosoma[i]/Suma
    return cromosoma

def PuntosIniciales():
    return int(np.random.uniform(1, tam))

class Individual:
    def __init__(self, color, cromosoma, x, y):
        self.cromosoma = CromosomaNormalizada(cromosoma)
        self.color = Color(self.cromosoma) # 'blue'
        self.x = x
        self.y = y
        self.fitness = 0
        self.steps = 0

    def evaluate_fitness(self, steps):
        return 1/steps if steps != 0 else 1 

    def mutate(self): #random.random = 0-1
        if random.random() < mutation_rate: 
            index = random.randint(0, len(self.cromosoma)-1)#pick a random index of the chromosome?
            #randomize said element chosen above
            self.cromosoma[index]  = random.random()
            self.cromosoma = CromosomaNormalizada(self.cromosoma)
        return

class Population:
    def __init__(self, N):
        self.individuals = []
        self.finishers = []
        occupied_positions = set()

        for _ in range(N):
            x, y = PuntosIniciales(), PuntosIniciales()

            # Generate new positions until we find an unoccupied one
            while (x, y) in occupied_positions:
                x, y = PuntosIniciales(), PuntosIniciales()

            occupied_positions.add((x, y))
            self.individuals.append(Individual(None, cromosoma=Cromosoma(), x=x, y=y))

        
    def move(self):
        remaining_individuals = []

        for individual in self.individuals: #move
            direccion = np.random.choice(range(len(Direcciones)), p=individual.cromosoma)
            new_x = individual.x + Direcciones[direccion][0] #[1,1]
            new_y = individual.y + Direcciones[direccion][1]

            if individual.x != tam: # Only move if they haven't reached the end
                if 1 <= new_x <= tam and 1 <= new_y <= tam: #check for collision
                    if not any(other_individual.x == new_x and other_individual.y == new_y for other_individual in self.individuals+self.finishers):
                        individual.x = new_x
                        individual.y = new_y

            # Check for goal-reached after potentially moving
            if individual.x == tam: 
                self.finishers.append(individual)
                individual.steps = _+1
                individual.fitness = individual.evaluate_fitness(individual.steps)
            else:
                remaining_individuals.append(individual)
                
        print(len(remaining_individuals),"remaining individuals at step",_)
        self.individuals = remaining_individuals


    def tournament_selection(self, k):
        best = None
        for _ in range(k):
            ind = random.choice(self.finishers)
            if (best is None) or ind.fitness > best.fitness:
                best = ind
        return best
    
    def tournament_but_epic(self, k):
        # Select k individuals from the population at random
        tournament = random.sample(self.finishers, k)
        # Sort the selected individuals by fitness (from highest to lowest)
        tournament.sort(key=lambda x: x.fitness, reverse=True)

        p = 0.3  # Define your selection probability here
        # Calculate the cumulative probabilities for the individuals
        cumulative_probabilities = [p * (1 - p) ** i for i in range(k)]
        cumulative_probabilities = np.cumsum(cumulative_probabilities)

        # Select a random number between 0 and 1
        r = random.random()
        # Find the individual whose cumulative probability bracket contains r
        for i in range(k):
            if r <= cumulative_probabilities[i]:
                print('torney-',tournament[i])
                return tournament[i]

    def crossover_but_epic(self, parent1, parent2):
        crossover_binary = Cromosoma()
        child1_cromosoma,child2_cromosoma = [],[]
        for i in range(len(crossover_binary)):
            crossover_binary[i] = int(crossover_binary[i])
            if i < 1: # if 0 take from parent 1, else take from parent 2
                print(parent1)
                
                child1_cromosoma.append(parent1.cromosoma[i])
                child2_cromosoma.append(parent2.cromosoma[i])
            else:
                child1_cromosoma.append(parent2.cromosoma[i])
                child2_cromosoma.append(parent1.cromosoma[i])
        child1_cromosoma, child2_cromosoma = CromosomaNormalizada(child1_cromosoma), CromosomaNormalizada(child2_cromosoma)

        child1 = Individual(None, cromosoma=child1_cromosoma, x=PuntosIniciales(), y=PuntosIniciales())
        child2 = Individual(None, cromosoma=child2_cromosoma, x=PuntosIniciales(), y=PuntosIniciales())

        return child1,child2  
       
    def crossover(self, parent1, parent2):
        # Select random crossover point
        crossover_point = np.random.randint(1, len(parent1.cromosoma))
        
        # Create children by swapping genes after the crossover point
        child1_cromosoma = np.concatenate((parent1.cromosoma[:crossover_point], 
                                            parent2.cromosoma[crossover_point:]))
        child2_cromosoma = np.concatenate((parent2.cromosoma[:crossover_point], 
                                            parent1.cromosoma[crossover_point:]))

        # Normalize the chromosomes
        child1_cromosoma = CromosomaNormalizada(child1_cromosoma)
        child2_cromosoma = CromosomaNormalizada(child2_cromosoma)

        # Create new Individual instances for children
        child1 = Individual(None, cromosoma=child1_cromosoma, x=PuntosIniciales(), y=PuntosIniciales())
        child2 = Individual(None, cromosoma=child2_cromosoma, x=PuntosIniciales(), y=PuntosIniciales())

        # Return children
        return child1, child2


    def create_new_generation(self):
        C = []
        C = self.finishers
        print(len(C))
        while len(C) < population_size:
            parent1 = self.tournament_selection(2)  # Binary tournament selection
            parent2 = self.tournament_selection(2)
            while parent1 == parent2:
                parent2 = self.tournament_selection(2)
            print('create_new',parent1)
            child1, child2 = self.crossover_but_epic(parent1, parent2)
            C.append(child1)
            C.append(child2)

        # Generate new starting points for all individuals in the new generation
        occupied_positions = set()
        for individual in C:
            individual.mutate()
            x, y = PuntosIniciales(), PuntosIniciales()

            # Generate new positions until we find an unoccupied one
            while (x, y) in occupied_positions:
                x, y = PuntosIniciales(), PuntosIniciales()

            individual.x = x
            individual.y = y
            occupied_positions.add((x, y))

        # Replace old population with new generation
        self.individuals = C
        self.finishers = []
        print(len(self.individuals))

population_size = 50
generations = 10
mutation_rate = 0.05
tam = 30
max_steps = int(population_size*1.5)

P = Population(population_size)
#print(P.individuals[1].cromosoma," : ",P.individuals[1].color)
#plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.ion()
sc = ax.scatter([individual.x for individual in P.individuals],
                [individual.y for individual in P.individuals],
                c = [individual.color for individual in P.individuals],
                marker='s', s=100)
plt.xlim(0,tam)
plt.ylim(0,tam)
plt.draw()

for i in range(generations):
    for _ in range(max_steps):
        P.move()
        # Update the positions drawn
        all_individuals = P.individuals + P.finishers
        sc.set_offsets([[individual.x, individual.y] for individual in all_individuals])
        sc.set_facecolor([individual.color for individual in all_individuals])
        # Redraw the plot
        fig.canvas.draw_idle()
        plt.pause(0.05)

    P.create_new_generation()
    all_individuals = P.individuals
    sc.set_offsets([[individual.x, individual.y] for individual in all_individuals])
    sc.set_facecolor([individual.color for individual in all_individuals])
    fig.canvas.draw_idle()
    # Add title to figure with current generation
    plt.title(f"Generación: {i+1}")
    plt.pause(2)

#P.create_new_generation
plt.waitforbuttonpress()