import numpy as np
from numpy.random import randn


import numpy as np
import math

from pyretis.core import system


from ParticleSwarmOptimization_EvolutionaryStrategy.function.settingFunction import evaluate_potential_grid, function_name


class EvolutionStrategy:

    def fitnessFunc(self, chromosome):
        """"""
        firstSum = 0.0
        secondSum = 0.0
        for c in chromosome:
            firstSum += c ** 2.0
            secondSum += math.cos(2.0 * math.pi * c)
        n = float(len(chromosome))
        X,Y,Z=evaluate_potential_grid(firstSum / n,secondSum / n)
        return Z


    def __init__(self, generations=50000, population_size=1, mutation_step=0.1, adjust_mutation_constant=0.8):
        self.generations = generations
        self.population_size = population_size

        self.cromossome = None
        self.mutation_step = mutation_step
        self.adjust_mutation_constant = adjust_mutation_constant
        self.success_rate = .2
        self.num_mutations = 0
        self.num_successful_mutations = 0
        self.verbose = 0

    def print_cromossome(self):
        print('[' + ','.join(["%.2f" % nb for nb in self.cromossome]) + ']')

    def init_cromossome(self):
        self.cromossome = 30*np.random.random(30)-30

    def get_mutation_vector(self):
        return np.random.normal(0, self.mutation_step, 30)

    def get_success_probability(self):
        return self.num_successful_mutations / float(self.num_mutations)

    def fitness(self, cromossome):
        return -1.*abs(self.fitnessFunc(cromossome))

    def adjust_mutation_step(self):
        ps = self.get_success_probability()
        if self.verbose == 1:
            print("ps: %.4f" % ps)
        if ps > self.success_rate:
            self.mutation_step /= self.adjust_mutation_constant
        elif ps < self.success_rate:
            self.mutation_step *= self.adjust_mutation_constant
        if self.verbose == 1:
            print("mutation_step: %.4f" % self.mutation_step)

    def apply_mutation(self):
        cromossome_prime = self.cromossome + self.get_mutation_vector()
        self.num_mutations += 1
        if self.fitness(self.cromossome) < self.fitness(cromossome_prime):
            self.cromossome = cromossome_prime
            self.num_successful_mutations += 1
        self.adjust_mutation_step()

    def run(self, verbose=0):
        self.temp=0.0
        self.counter=0
        self.verbose = verbose
        self.init_cromossome()
        gen = 0
        # history = [(self.cromossome, self.fitnessFunc(self.cromossome))]
        if self.verbose == 1:
            print("gen: %d" % gen)
            self.print_cromossome()
            # print("Ackley(x): %.5f" % self.fitnessFunc(self.cromossome))

            # print("Ackley(x): %.5f" % easom_potential(self.cromossome))
        while gen < self.generations:
            gen += 1
            self.apply_mutation()
            if self.verbose == 1:
                print("gen: %d" % gen)
                self.print_cromossome()
                print(str(function_name())+"(x): %.5f" % self.fitnessFunc(self.cromossome))
            # history.append((self.cromossome, self.fitnessFunc(self.cromossome)))
            if(float("%.5f" %self.fitnessFunc(self.cromossome))==0.0):
                return float("%.5f" %self.fitnessFunc(self.cromossome)),self.cromossome
            if(self.temp==float("%.5f" %self.fitnessFunc(self.cromossome))):
                self.counter+=1
            else:
                self.counter=0

            if(self.counter>10000):
                return  float("%.5f" %self.fitnessFunc(self.cromossome)),self.cromossome
            self.temp=float("%.5f" %self.fitnessFunc(self.cromossome))
        # return history

def naive_variance(data):
    n = 0
    Sum = 0
    Sum_sqr = 0

    for x in data:
        n = n + 1
        Sum = Sum + x
        Sum_sqr = Sum_sqr + x * x

    variance = (Sum_sqr - (Sum * Sum) / n) / (n - 1)
    return variance

def average(date):
    Sum=0
    for x in date:
        Sum+=x
    return  Sum/len(date)


arrFitness=[]
arrCromossome=[]

step=30
for i in range(step):
    es = EvolutionStrategy()
    fitness,cromossome=es.run(verbose = 1)
    arrFitness.append(fitness)
    arrCromossome.append(cromossome)

print("==================================================================")
for i in range(step):
    print('step'+str(i+1))
    print('cromossome: '+str(arrCromossome[i]))
    print('bestFitness: '+ str(arrFitness[i]))
print("==================================================================")
print(arrFitness)
print('Variance: '+str(naive_variance(arrFitness)))
print('Average: '+str(average(arrFitness)))