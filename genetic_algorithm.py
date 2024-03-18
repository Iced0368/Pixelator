from typing import TypeVar, List, Callable, Optional
import matplotlib.pyplot as plt
import random
import time

Data = TypeVar("Data")

class GeneticModel:
    def __init__(
            self,
            fitness:        Callable[[Data], float],
            crossover:      Callable[[Data, Data], Data],
            mutation:       Callable[[Data], Data],
            roulette:       Optional[Callable[[float], float]] = None,
    ):
        self.fitness = fitness
        self.crossover = crossover
        self.mutation = mutation
        self.roulette = roulette

    def run(
            self,
            initialize: List[Data],
            maximize: bool,
            selections: int, 
            crossovers: int, 
            mutations: int,
            generations: int,
            log: bool = False,
            show_graph: bool = False,
            fitness_goal: Optional[float] = None,
            returns: int = 1,
            convergent: int = -1, 
    ):
        s_time = time.time()

        current_generation = initialize.copy()
        current_fitness = [self.fitness(d) for d in current_generation]
        fitness_log = [max(current_fitness)]
        gens = 0
        no_evol = 0
        peak = 0
        result = initialize.copy()

        for i in range(generations):
            gens += 1
            if log:
                print(f'generation {i}:')
                #print(current_generation)
                #for gen in current_generation:
                #    print(gen)
            selected = random.choices(
                population=current_generation,
                weights=current_fitness if self.roulette is None else [self.roulette(f) for f in current_fitness],
                k=selections
            )
            crossedover = [
                self.crossover(random.choice(current_generation), random.choice(current_generation)) 
                for _ in range(crossovers)
            ]
            mutated = [
                self.mutation(random.choice(current_generation)) 
                for _ in range(mutations)
            ]
            current_generation = selected + crossedover + mutated
            random.shuffle(current_generation)
            current_fitness = [self.fitness(d) for d in current_generation]
            max_fitness = max(current_fitness) if maximize else min(current_fitness)
            fitness_log.append(max_fitness)

            result = sorted(result + current_generation, key=lambda x: -self.fitness(x) if maximize else self.fitness(x))[:returns]

            if fitness_goal is not None and (max_fitness >= fitness_goal and maximize or max_fitness <= fitness_goal and not maximize):
                break

            if max_fitness > peak and maximize:
                peak = max_fitness
                no_evol = 0
            if max_fitness < peak and not maximize:
                peak = max_fitness
                no_evol = 0
                
            else:
                no_evol += 1

            if no_evol == convergent:
                break

        if log:
            print(f'generation {gens}:')
            #print(current_generation)
            #for gen in current_generation:
            #    print(gen)
            print(f'time spent: {time.time() - s_time}')
        if show_graph:
            plt.plot(fitness_log)
            plt.xlabel('gens')
            plt.ylabel('fitness')
            plt.show()

        return result

