from fuzzy_input import FuzzyInput
from fuzzy_output import FuzzyOutput
from fuzzy_rule import FuzzyRule
from fuzzy_logic_operator import FuzzyLogicOperator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
from math import isnan
from tqdm import tqdm
from copy import deepcopy

class FuzzyInputColumn:
    def __init__(self, fuzzy_inputs: list):
        self.fuzzy_inputs = fuzzy_inputs
        self.input_count = len(fuzzy_inputs)

class FuzzyOutputColumn:
    def __init__(self, fuzzy_outputs: list):
        self.fuzzy_outputs = fuzzy_outputs
        self.output_count = len(fuzzy_outputs)

class FuzzyIndividual:
    def __init__(self, initial_chromosome):
        self.chromosome = initial_chromosome
        self.fitness = float('inf')
       

class FuzzyPopulationController:
    # encode 2 inputs to 1 output
    # custom encoding functions for multi-output must be implemented by hand currently
    # custom encoidng functions for multi-input are not supported because a fuzzy logic input of more than 2 gates
    # can be described as multiple composites of 2 rule factors
    @staticmethod
    def encode(id1, fuzzy_input_1, id2, fuzzy_input_2, id3, fuzzy_output_1, fuzzy_operator):
        return [id1, fuzzy_input_1, id2, fuzzy_input_2, id3, fuzzy_output_1, fuzzy_operator]

    # evaluation function is x1 - potrosnja
    #                        x2 - pouzdanost
    # for other datasets, must manually select points of interest
    def evaluate(self, chromo, i1, i2, o1):
        chromosome = deepcopy(chromo)
        # print("\033[34mZa potrosnju {} i pouzdanost {} vrednost je {}\033[0m".format(i1, i2, o1))
        assert(len(chromosome) % 7 == 0)        
        gene_count = int(len(chromosome) / 7)

        for i in range(gene_count):
            id1 = chromosome[i * 7]
            fuzzy_input_1 = chromosome[i*7 + 1]
            id2 = chromosome[i*7 + 2]
            fuzzy_input_2 = chromosome[i*7 + 3]
            id3 = chromosome[i*7 + 4]
            fuzzy_output_1 = chromosome[i*7 + 5]
            fuzzy_operator = chromosome[i*7 + 6]

            op1:FuzzyInput = self.fuzzy_input_columns[id1].fuzzy_inputs[fuzzy_input_1]
            op2:FuzzyInput = self.fuzzy_input_columns[id2].fuzzy_inputs[fuzzy_input_2]
            op3:FuzzyOutput = self.fuzzy_output_columns[id3].fuzzy_outputs[fuzzy_output_1]

            if (id1 == 0): #potrosnja
                op1.setX(i1)
            elif(id1 == 1): #pouzdanost
                op1.setX(i2)
            else:
                print("error -> no value has been set on op1 : check your dataset configuration")

            if (id2 == 0):
                op2.setX(i1)
            elif(id2 == 1):
                op2.setX(i2)
            else:
                print("error -> no value has been set on op2 : check your dataset configuration")

            if (i == 0):
                op3.mu = 0

            if (fuzzy_operator == 0):
                fuzzy_operator = FuzzyLogicOperator.AND
            else: 
                fuzzy_operator = FuzzyLogicOperator.OR

            FuzzyRule(operand1=op1, operand2=op2, output=op3, operator=fuzzy_operator)

        final_value = self.defuzzify()
        op3.mu = 0
        
        if (isnan(final_value)):
            return float('inf')
            
        return (o1 - final_value)**2 

    def __init__(self) -> None:
        potrosnja = []
        potrosnja.append(FuzzyInput('mala potrosnja',     [3,10],         [1,0],              0))
        potrosnja.append(FuzzyInput('srednja potrosnja',  [7,10,12,15],   [0,1,1,0],          0))
        potrosnja.append(FuzzyInput('velika potrosnja',   [12,15],        [0,1],              0))

        pouzdanost = []
        pouzdanost.append(FuzzyInput('visoka pouzdanost',  [5,10],         [1,0],             0))
        pouzdanost.append(FuzzyInput('niska pouzdanost',   [8,15],         [0,1],             0))

        vrednost = []
        vrednost.append(FuzzyOutput('mala vrednost',     [7,15],         [1,0]                ))
        vrednost.append(FuzzyOutput('srednja vrednost',  [7,15,25,40],   [0,1,1,0]            ))
        vrednost.append(FuzzyOutput('velika vrednost',   [25,40],        [0,1]                ))
        
        self.vrednost = vrednost

        ci1 = FuzzyInputColumn(potrosnja)
        ci2 = FuzzyInputColumn(pouzdanost)
        co1 = FuzzyOutputColumn(vrednost)

        fuzzy_input_columns = list()
        fuzzy_output_columns = list()

        fuzzy_input_columns.append(ci1)
        fuzzy_input_columns.append(ci2)
        fuzzy_output_columns.append(co1)

        self.fuzzy_input_columns = fuzzy_input_columns
        self.fuzzy_output_columns = fuzzy_output_columns
        
    def defuzzify(self):
        cs = np.array([v.c for v in self.vrednost])
        mus = np.array([v.mu for v in self.vrednost])
        
        if mus.sum() == 0:
            return float('inf')
        return cs.dot(mus) / mus.sum()


    # creates a new string representation of a chromosome
    def create_random_fuzzy_gene(self) -> FuzzyIndividual:
        column_id_1 = random.randrange(0, len(self.fuzzy_input_columns))
        column_id_2 = -1
        
        while (True):
            column_id_2 = random.randrange(0, len(self.fuzzy_input_columns))
            if (column_id_1 != column_id_2):
                break

        fuzzy_input_1 = random.randrange(len(self.fuzzy_input_columns[column_id_1].fuzzy_inputs))
        fuzzy_input_2 = random.randrange(len(self.fuzzy_input_columns[column_id_2].fuzzy_inputs))

        column_id_3 = random.randrange(0, len(self.fuzzy_output_columns))
        fuzzy_output_1 = random.randrange(0, len(self.fuzzy_output_columns[column_id_3].fuzzy_outputs))
        fuzzy_operator = random.randrange(0, 2)

        new_chromosome = self.encode(column_id_1, fuzzy_input_1, column_id_2, fuzzy_input_2, column_id_3, fuzzy_output_1, fuzzy_operator)
        return new_chromosome

class GeneticAlgorithm:
    def __init__(self, controller: FuzzyPopulationController, population_size:int, train_dataframe:pd.DataFrame, tournament_size :int,  mutation_chance:float, elitism,  epochs=2000) -> None:
        self.population = [FuzzyIndividual(controller.create_random_fuzzy_gene()) for _ in range (population_size)]
        self.controller = controller
        self.epochs = epochs
        self.train_dataframe = train_dataframe
        self.mutation_chance = mutation_chance
        self.elitism = elitism
        self.tournament_size = tournament_size

    def print_rules(self, rules : FuzzyIndividual):
        genes_number = int(len(rules.chromosome)/7)
        for i in range (genes_number):
            id1 = rules.chromosome[i * 7]
            fuzzy_input_1 = rules.chromosome[i*7 + 1]
            id2 = rules.chromosome[i*7 + 2]
            fuzzy_input_2 = rules.chromosome[i*7 + 3]
            id3 = rules.chromosome[i*7 + 4]
            fuzzy_output_1 = rules.chromosome[i*7 + 5]
            fuzzy_operator = rules.chromosome[i*7 + 6]

            op1:FuzzyInput = self.controller.fuzzy_input_columns[id1].fuzzy_inputs[fuzzy_input_1]
            op2:FuzzyInput = self.controller.fuzzy_input_columns[id2].fuzzy_inputs[fuzzy_input_2]
            op3:FuzzyOutput = self.controller.fuzzy_output_columns[id3].fuzzy_outputs[fuzzy_output_1]
            op = 'AND'
            if fuzzy_operator == 1:
                op = 'OR'

            print (f'{op1.name} {op} {op2.name} => {op3.name}')

    def mutation(self, individual : FuzzyIndividual):
    
        chromosome = individual.chromosome
        new_chromosome = deepcopy(chromosome)
        # moguce mutacije - dodavanje pravila, brisanje, izmena
    
        for mutation in ["delete", "add", "update"]:
            pm = random.random()
            if (pm > self.mutation_chance):
                continue

            if (mutation == "add"):
                to_add = self.controller.create_random_fuzzy_gene()
                number_of_genes = int(len(new_chromosome)/7)
                genes = []
                for i in range(number_of_genes):
                    genes.append(new_chromosome[i*7 : (i+1)*7])

                if to_add not in genes:
                    new_chromosome += to_add
            
            elif (mutation == "delete" and len(chromosome) > 7):
                number_of_genes = int(len(chromosome) / 7)
                selected_gene = random.randrange(0, number_of_genes)

                new_chromosome = chromosome[:(selected_gene*7)] + chromosome[(selected_gene*7+7):]

            else: # update or len(chromosome) = 7 
                number_of_genes = int(len(new_chromosome) / 7)
                selected_gene = random.randrange(0, number_of_genes)    # gene selected

                id1 = new_chromosome[selected_gene * 7]                 # cannot mutate
                id2 = new_chromosome[selected_gene * 7 + 2]               # cannot mutate
                id3 = new_chromosome[selected_gene * 7 + 4]               # cannot mutate

                fuzzy_operator = new_chromosome[selected_gene * 7 + 6]

                for op in ['left input','right input', 'output', 'operator']:
                    p = random.random()
                    if (p > self.mutation_chance):
                        continue
                    if (op == 'left input'):
                        possible_inputs = len(self.controller.fuzzy_input_columns[id1].fuzzy_inputs)
                        new_chromosome[selected_gene * 7 + 1] = random.randrange(0, possible_inputs)
                    elif (op == 'right input'): 
                        possible_inputs = len(self.controller.fuzzy_input_columns[id2].fuzzy_inputs)
                        new_chromosome[selected_gene * 7 + 3] = random.randrange(0, possible_inputs)
                    elif (op == 'output'):
                        possible_outputs = len(self.controller.fuzzy_output_columns[id3].fuzzy_outputs)
                        new_chromosome[selected_gene * 7 + 5] = random.randrange(0, possible_outputs)
                    elif (op == 'operator'):
                        if (fuzzy_operator == 0):
                            new_chromosome[selected_gene * 7 + 6] = 1
                        else:
                            new_chromosome[selected_gene * 7 + 6] = 0 
        individual.chromosome = new_chromosome       
    
    def selection(self):
        tournament = random.sample(self.population, self.tournament_size)
        return min(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1 : FuzzyIndividual, parent2 : FuzzyIndividual, child1 : FuzzyIndividual, child2 : FuzzyIndividual):
        ch1 = parent1.chromosome
        ch2 = parent2.chromosome


        random_gene = random.randrange(0, int(len(ch1) / 7)) 
        if len(ch2) < len(ch1):
            random_gene = random.randrange(0, int(len(ch2) / 7)) 

        child1_chromosome = parent1.chromosome[:7*random_gene] + parent2.chromosome[7*random_gene:]
        child2_chromosome = parent2.chromosome[:7*random_gene] + parent1.chromosome[7*random_gene:]

        child1.chromosome = child1_chromosome
        child2.chromosome = child2_chromosome
        
                
    def run(self):

        #evaluacija inicijalnih
        for i in range (len(self.population)):
            individual = self.population[i]
            self.population[i].fitness = self.calc_fitness(individual.chromosome)
        new_population = self.population.copy()
        
        for i in range(self.epochs):
            self.population.sort(key=lambda x: x.fitness)
            new_population[:self.elitism] = self.population[:self.elitism]

            for j in range(self.elitism, len(self.population)-1, 2):
                parent1 = self.selection() 
                parent2 = self.selection()  
                while (parent1 == parent2):
                    parent2 = self.selection()  

                self.crossover(parent1, parent2, child1=new_population[j], child2=new_population[j+1])

                self.mutation(individual=new_population[j])
                self.mutation(individual=new_population[j+1])
            
                new_population[j].fitness = self.calc_fitness(new_population[j].chromosome)
                new_population[j+1].fitness = self.calc_fitness(new_population[j+1].chromosome)
        
            self.population = new_population

            print (f'Epoha {i+1} fitness {min(self.population, key=lambda x: x.fitness).fitness} best {min(self.population, key=lambda x: x.fitness).chromosome}')
        self.print_rules (min(self.population, key=lambda x: x.fitness))
    
    def calc_fitness(self, chromosome):
        squared_error = 0
        for i in range(len(self.train_dataframe)):
            i1,i2,o1 = self.train_dataframe.at[i, "potrosnja"], self.train_dataframe.at[i, "pouzdanost"], self.train_dataframe.at[i, "vrednost"]
            squared_error += self.controller.evaluate(chromo=chromosome, i1=i1, i2=i2, o1=o1)
        return squared_error


if __name__ == "__main__":

    
    df = pd.read_csv('3.csv')
    print(df.head)
    fpc = FuzzyPopulationController()
    g1 = GeneticAlgorithm(population_size=50, train_dataframe=df, controller=fpc, tournament_size= 5, mutation_chance=0.3, elitism=5,  epochs=300)
    g1.run()

