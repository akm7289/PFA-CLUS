"""PFA Clustering
Optimizing centroid using K-Means style. In hybrid mode will use K-Means to seed first particle's centroid
"""
import math

import numpy as np

import kmeans
from kmeans import calc_sse
from particle import Particle
import numpy as np
from copy import deepcopy
from math import gamma, pi
from models.multiple_solution.root_multiple import RootAlgo
from constants import *



class OPFAClustering(RootAlgo):
    ID_POS = 0
    ID_FIT = 1

    def _create_solution__(self, minmax=0):
        solution = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)

        return [solution, 1.5]

    def cs(self, i):
        solution = self.particles[i].position
        fitness = self.particles[i].current_score
        return [solution, fitness]

    def _fitness__(self, solution):
        particle = Particle(solution=solution, type=3, fitness_function=self.fitness_function)
        return particle.get_score(self.data)

    def _fitness_model__(self, solution=None, minmax=0):
        particle = Particle(solution=solution, type=3, fitness_function=self.fitness_function)
        return particle.get_score(self.data)

    def __init__(self,
                 n_cluster: int,
                 n_particles: int,
                 data: np.ndarray,
                 initialization_help: int = 0,
                 max_iter: int = 1000,
                 print_debug: int = 10,
                 alpha_limit=.3,
                 beta_limit=.8,
                 fitness_function=0,
                 use_kmeans_after=False):
        self.domain_range = [np.min(data), np.max(data)]
        self.problem_size = data.shape[1]
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.pop_size = n_particles
        self.epoch = max_iter
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.initialization_help = initialization_help
        self.print_debug = print_debug
        self.feasible_domains=self.compute_feasible_dimenstions(data)
        self.fitness_function=fitness_function
        self.use_kmeans_after=use_kmeans_after


        self.pathfinder_score = np.inf
        self.pathfinder_centroids = None
        self.pathfinder_sse = np.inf
        self._init_particles()
        self.alpha_limit=alpha_limit
        self.beta_limit=beta_limit
        self.particles=sorted(self.particles, key=lambda temp: temp.current_score)




    def _init_particles(self):
        for i in range(self.n_particles):
            particle = None
            if self.initialization_help == RANDOM_NUMBER:
                solution = np.random.uniform(self.domain_range[0], self.domain_range[1],
                                             size=(self.n_cluster, self.problem_size))
                particle = Particle(solution=solution, type=3, fitness_function=self.fitness_function)
                particle.current_score = particle.get_score(self.data)
                particle.current_sse = calc_sse(solution, particle._predict(self.data), self.data)

            elif (i == 0) and (self.initialization_help == USE_KMEANS or self.initialization_help==HYPRID):
                particle = Particle(n_cluster=self.n_cluster, data=self.data,
                                    initialization_help=USE_KMEANS, fitness_function=self.fitness_function,max_iterations=self.max_iter)
            else:
                particle = Particle(n_cluster=self.n_cluster, data=self.data,
                                    initialization_help=self.initialization_help, fitness_function=self.fitness_function)
            if particle.current_score < self.pathfinder_score:
                self.pathfinder_centroids = particle.centroids.copy()
                self.pathfinder_score = particle.current_score
            self.particles.append(particle)
            self.pathfinder_sse = min(particle.current_sse, self.pathfinder_sse)
        return None

    def _amend_solution_and_return__(self, solution=None):
        for c in range(self.n_cluster):
            for i in range(self.problem_size):
                if solution[c][i] < self.feasible_domains[0][i]:
                    solution[c][i] = self.feasible_domains[0][i]
                if solution[c][i] > self.feasible_domains[1][i]:
                    solution[c][i] =self.feasible_domains[1][i]
        return solution
    def _amend_solution_and_return2__(self, solution=None):
        for c in range(self.n_cluster):
            for i in range(self.problem_size):
                if solution[c][i] < self.feasible_domains[0][i]:
                    solution[c][i] = self.feasible_domains[0][i]+abs(self.feasible_domains[0][i]-solution[c][i])
                if solution[c][i] > self.feasible_domains[1][i]:
                    solution[c][i] =self.feasible_domains[1][i]-abs(self.feasible_domains[1][i]-solution[c][i])
        return solution


    def _test_boundry__(self, solution=None):
        for c in range(self.n_cluster):
            for i in range(self.problem_size):
                if solution[c][i] < self.feasible_domains[0][i]:
                    return True
                if solution[c][i] > self.feasible_domains[1][i]:
                    return True
        return False

    def run(self):

        print('Initial Pathfinder score', self.pathfinder_score)
        # Init pop and calculate fitness
        self.print_train = self.print_debug>0  # self.objective_func = root_algo_paras["objective_func"]
        self.epoch = self.max_iter

        pop = sorted(self.particles, key=lambda temp: temp.current_score)
        g_best, gbest_present = None, None
        g_best = deepcopy(pop[0])
        gbest_present = deepcopy(g_best)

        # Find the pathfinder
        out_of_boundry_counter=0
        oppositionCount=0
        history = []
        alphaLimit=self.alpha_limit
        betaLimit=self.beta_limit
        counter=0
        sumOfSquareErrorsList=[]
        for i in range(self.epoch):
            alpha=np.random.uniform(0, alphaLimit)
            beta=np.random.uniform(0, betaLimit)

            if i%10==0:
                counter += 1
                alphaLimit = alphaLimit * math.pow(.9,counter)
                betaLimit = betaLimit * 1.1

            A = np.random.uniform(-.1, .1) * np.exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present.centroids + 2 * np.random.uniform(0,.1) * (
                    gbest_present.centroids - g_best.centroids) + A
            if self._test_boundry__(temp):
                pass
                # print("^"*100)
                # print("Lead out of boundry")
                # print("alpha:",alpha)
                # print("beta:",beta)
                # print("^"*100)
            #temp = self._amend_solution_and_return2__(temp)
            temp = self._amend_solution_and_return__(temp)
            fit = self._fitness__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present.current_score:
                gbest_present = Particle(solution=temp, fitness=fit, type=2,fitness_function=self.fitness_function)
                pop[0] = deepcopy(gbest_present)




            ## Update positions of members, check the bound and calculate new fitness
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop[j].centroids)
                temp2 = deepcopy(pop[j].centroids)

                t1 = beta * np.random.uniform(0,.1) * (gbest_present.centroids - temp1)
                for k in range(1, self.pop_size):
                    # dist = np.linalg.norm(pop[k].centroids - temp1)
                    dist = pop[k].centroids - temp1
                    t2 = alpha * np.random.uniform(0,.1) * (pop[k].centroids - temp1)
                    t3 = np.random.uniform(-.1, .1, self.problem_size) * (
                            1 - (i + 1) * 1.0 / self.epoch) * dist
                    # t3 = np.random.uniform(-1,1, self.problem_size) * (
                    #             1 - (i + 1) * 1.0 / self.epoch) * dist
                    temp2 += t2 + t3
                temp2 += t1

                ## Update members
                if self._test_boundry__(temp2):
                    out_of_boundry_counter+=1
                #temp2 = self._amend_solution_and_return2__(temp2)
                temp2 = self._amend_solution_and_return__(temp2)
                fit = self._fitness__(temp2)
                if fit < pop[j].current_score:
                    pop[j] = Particle(solution=temp2, fitness=fit, type=2,fitness_function=self.fitness_function)
                else:
                    C_op = self.feasible_domains[1] * np.ones(self.problem_size) + self.feasible_domains[0] *  np.ones(self.problem_size) - temp2
                    #C_op=self._amend_solution_and_return2__(C_op)
                    C_op = self._amend_solution_and_return__(C_op)

                    fit_op = self._fitness_model__(C_op)
                    if fit_op < pop[j].current_score:
                        oppositionCount+=1
                        pop[j] = Particle(solution=C_op, fitness=fit_op, type=2,fitness_function=self.fitness_function)

            ## Update the best solution found so far (current pathfinder)
            current_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROBLEM)
            if current_best.current_score < gbest_present.current_score:
                gbest_present = deepcopy(current_best)

            sse=kmeans.calc_sse(gbest_present.centroids, gbest_present._predict(self.data), self.data)
            sumOfSquareErrorsList.append(sse)
            if self.print_train:
                print("Generation : {0}, best result so far: {1}".format(i + 1, gbest_present.current_score))

        print("out of boundry counter: ", out_of_boundry_counter)
        print("Opposite:", oppositionCount)

        self.particles = sorted(pop, key=lambda temp: temp.current_score)
        if self.use_kmeans_after:
            km=kmeans.KMeans(n_cluster=self.n_cluster,max_iter=self.epoch)
            km.fit(self.data, self.particles[0].centroids)
            self.particles[0].centroids = km.centroid.copy()
            kmeans_sol=km.centroid.copy()
            fit = self._fitness__(kmeans_sol)
            if fit < pop[0].current_score:
                pop[0] = Particle(solution=kmeans_sol, fitness=fit, type=2, fitness_function=self.fitness_function)
                print("fitness enhanced by kmeans")
            print(fit)
            gbest_present.current_score = fit





        return gbest_present.current_score, 1,sumOfSquareErrorsList,oppositionCount
    def _get_global_best__(self, pop=None, id_fitness=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp.current_score)
        return deepcopy(sorted_pop[0])

    def compute_feasible_dimenstions(self, data):
        min = data.min(axis=0)
        max = data.max(axis=0)
        return np.concatenate((min, max)).reshape((2, self.problem_size))


if __name__ == "__main__":
    pass








