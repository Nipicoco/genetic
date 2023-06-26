from random import randint, sample, random, uniform, seed
from argparse import ArgumentParser
from sys import exit
from time import strftime, localtime
from subprocess import call


parameter_settings = []
extfit_settings = {}
algorithm_settings = {}
verbose = False


def setup_arguments():
    """Take command-line arguments and read the input file"""
    # Argument parser for command line arguments
    parser = ArgumentParser(description="Genetic Algorithm Tournament Script")
    parser.add_argument('-v', '--verbose', action='store_true', help="Display extra information when running")
    parser.add_argument('input', help="Input file")
    # parser.add_argument('output', help="Output file")
    args = parser.parse_args()
    print("Genetic Algorithm Tournament Script")
    print("Start: %s" % strftime("%d %b %Y %H:%M:%S %z\n", localtime()))
    # Set verbosity
    global verbose
    verbose = args.verbose
    # Read input file and assign to settings dictionary
    print("Reading input from %s" % args.input)
    reading_type = 0
    try:
        with open(args.input) as file:
            for line in file:
                line = line.strip()
                if line == '[PARAMETERS]':
                    reading_type = 1
                elif line == '[EXTFIT]':
                    reading_type = 2
                elif line == '[ALGORITHM]':
                    reading_type = 3
                elif not line.startswith('#') and line:  # Ignore comments and blank lines
                    # Add parameter ranges, if a single number it's not going to mutate/change
                    if reading_type == 1:
                        if line.startswith('1'):
                            (_, min, max) = line.split()
                            parameter_settings.append([float(min), float(max)])
                        elif line.startswith('0'):
                            (_, value) = line.split()
                            parameter_settings.append(float(value))
                    elif reading_type == 2:
                        (key, val) = line.split()
                        extfit_settings[key] = val
                    # Read algorithm settings
                    elif reading_type == 3:
                        (key, val) = line.split()
                        if key == 'seed':
                            algorithm_settings[key] = val
                        else:
                            algorithm_settings[key] = float(val)
    except IOError:
        exit("File not found")
    except ValueError:
        exit("Input file formatted incorrectly")
    print("Parameter Settings: %s" % parameter_settings)
    print("External Fitness Settings: %s" % extfit_settings)
    print("Algorithm Settings: %s\n" % algorithm_settings)
    # Check to see that all required settings were given to script
    required_algorithm_settings = ['population_size', 'convergence_criterion', 'crossover_probability',
                                   'mutation_probability', 'tournament_size', 'tournament_size_losers',
                                   'max_generations', 'seed']
    for required_algorithm_setting in required_algorithm_settings:
        if required_algorithm_setting not in algorithm_settings:
            exit("Not all required algorithm settings found in input file")
    # Check to see that the extfit settings were given correctly
    required_extfit_settings = ['fitness_mode', 'external_write', 'external_script', 'external_read']
    if 'fitness_mode' not in extfit_settings:
        exit("The parameter fitness_mode was not found")
    if extfit_settings['fitness_mode'] == 'internal':
        pass
    elif extfit_settings['fitness_mode'] == 'external':
        for required_extfit_setting in required_extfit_settings:
            if required_extfit_setting not in extfit_settings:
                exit("Not all required external fitness settings found in input file")
    else:
        exit("The parameter fitness_mode was not set to 'internal' or 'external'")
    # Check to see that at least one thing can be optimized
    contains_at_least_one_modifiable = False
    for gene in parameter_settings:
        if type(gene) is list and gene[0] != gene[1]:
            contains_at_least_one_modifiable = True
    if not contains_at_least_one_modifiable:
        exit("You need at least one modifiable value to run a genetic algorithm!")


def run():
    """Run the genetic algorithm"""
    # Set the random seed
    seed(algorithm_settings['seed'])
    # Set up initial population
    generation = 0
    population = Population(int(algorithm_settings['population_size']))
    population.update_all_fitness()
    # Find current fittest
    most_fit = population.get_most_fit()
    print("Generation: %d | Most Fit: %s | Fitness: %f\n" % (generation, most_fit.chromosome, most_fit.fitness))
    # Loop
    while most_fit.fitness > algorithm_settings['convergence_criterion']\
            and generation < int(algorithm_settings['max_generations']):
        generation += 1
        population.selection()  # Could use alternate selections when close to converging
        if random() < algorithm_settings['crossover_probability']:  # Probability of crossover
            offspring = population.crossover()
            if random() < algorithm_settings['mutation_probability']\
                    or population.fit_one.chromosome == population.fit_two.chromosome:  # Probability of mutation
                offspring.mutation()
            population.find_not_fit()
            population.not_fit.chromosome = offspring.chromosome
        population.update_all_fitness()
        most_fit = population.get_most_fit()
        print("Generation: %d | Most Fit: %s | Fitness: %f%s" % (generation, most_fit.chromosome, most_fit.fitness,
                                                                 '\n' if verbose else ''))
    if generation >= int(algorithm_settings['max_generations']):
        print("\nStop on Generation: %d | Chromosome: %s" % (generation, most_fit.chromosome))
    else:
        print("\nConverged on Generation: %d | Chromosome: %s" % (generation, most_fit.chromosome))
    print("End: %s" % strftime("%d %b %Y %H:%M:%S %z", localtime()))


def internal_fitness(chromosome):
    """Determine the fitness of a particular chromosome and return its fitness value"""
    # Define a custom fitness function here!
    score = ((chromosome[0] - 0.1) ** 2 + (chromosome[1] - 1.5) ** 2 + (chromosome[2] - 3.0) ** 2
             + (chromosome[3] - 4.44) ** 2 + (chromosome[4] - 5.0) ** 2 + (chromosome[5] - 10.0) ** 2)
    return score


def external_fitness(population):
    # Write parameters to a file, one chromosome per line
    with open(extfit_settings['external_write'], 'w') as write_file:
        for individual in population:
            write_file.write('  '.join('%12.6f' % gene for gene in individual.chromosome) + '\n')  # Python OMG
    # Call the Perl script and wait for it to finish
    call('perl ' + extfit_settings['external_script'])
    # Read the output and update fitness values
    with open(extfit_settings['external_read']) as read_file:
        for num, line in enumerate(read_file):  # enumerate gives us a counter as well
            population[num].fitness = float(line)


class Population:
    """Represents a population with multiple chromosomes advancing through generations"""

    def __init__(self, population_size):
        self.population = []
        if verbose:
            print("Creating population of size %d" % population_size)
        for index in range(0, population_size):
            individual = Individual()
            if verbose:
                print("Index: %d | Chromosome: %s | Fitness: %f" % (index, individual.chromosome, individual.fitness))
            self.population.append(individual)
        self.fit_one = None
        self.fit_two = None
        self.not_fit = None
        print('')

    def update_all_fitness(self):
        """Update fitness for all chromosomes in population"""
        if extfit_settings['fitness_mode'] == 'internal':
            for individual in self.population:
                individual.fitness = internal_fitness(individual.chromosome)
        else:
            external_fitness(self.population)

    def get_most_fit(self):
        """Return the fittest chromosome in the population"""
        return min(self.population, key=lambda individual: individual.fitness)

    def selection(self):
        """Update the two parent chromosomes by way of tournament selection"""
        # Tournament Selection
        tournament_one = sample(self.population, int(algorithm_settings['tournament_size']))
        tournament_two = sample(self.population, int(algorithm_settings['tournament_size']))
        # Find which is the most fit in each tournament
        self.fit_one = min(tournament_one, key=lambda individual: individual.fitness)
        self.fit_two = min(tournament_two, key=lambda individual: individual.fitness)
        if verbose:
            print("Parent ONE Chromosome: %s | Fitness: %f" % (self.fit_one.chromosome, self.fit_one.fitness))
            print("Parent TWO Chromosome: %s | Fitness: %f" % (self.fit_two.chromosome, self.fit_two.fitness))

    def find_not_fit(self):
        """Update the not-fit chromosome to be removed by tournament selection"""
        # Tournament Selection
        tournament_not_fit = sample(self.population, int(algorithm_settings['tournament_size_losers']))
        # Find which is the most fit in each tournament
        self.not_fit = max(tournament_not_fit, key=lambda individual: individual.fitness)
        if verbose:
            print("Unfit chromosome to be replaced: %s | Fitness %f" % (self.not_fit.chromosome, self.not_fit.fitness))

    def crossover(self):
        """Make a new individual and set its chromosome by averaging the parents (two fit individuals)"""
        baby = Individual()
        for index in range(0, len(baby.chromosome)):
            baby.chromosome[index] = (self.fit_one.chromosome[index] + self.fit_two.chromosome[index]) / 2
        if verbose:
            print("New chromosome: %s" % baby.chromosome)
        return baby


class Individual:
    """An individual chromosome"""

    def __init__(self):
        # Randomly initialize chromosome
        self.chromosome = []
        for gene in range(0, len(parameter_settings)):
            if type(parameter_settings[gene]) is float:
                self.chromosome.append(parameter_settings[gene])
            else:
                self.chromosome.append(uniform(parameter_settings[gene][0], parameter_settings[gene][1]))
        self.fitness = 0

    def mutation(self):
        """Mutate at a random mutationable index"""
        mutate_index = randint(0, len(self.chromosome) - 1)
        while not type(parameter_settings[mutate_index]) is list:  # Keep trying to find a mutationable index
            mutate_index = randint(0, len(self.chromosome) - 1)
        self.chromosome[mutate_index] = uniform(parameter_settings[mutate_index][0], parameter_settings[mutate_index][1])
        if verbose:
            print("Mutated chromosome: %s" % self.chromosome)


if __name__ == '__main__':
    setup_arguments()
    run()
