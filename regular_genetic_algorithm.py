from genetic_algorithm import GeneticAlgorithm
import numpy as np
import random
import time


class RegularGeneticAlgorithm(GeneticAlgorithm):

    def solve(self, matrices, constraints_list, immutable_positions):
        elite_percentage = 5
        num_elites = elite_percentage * len(matrices) // 100
        generations = 0
        self.max_score = len(constraints_list) + len(immutable_positions) + 2 * (self.board_size) * (
                    self.board_size - 1)
        scores = dict()
        print(f"desired score: {self.max_score}")
        print()
        time.sleep(1)
        while generations < 20000:
            for i, matrix in enumerate(matrices):
                scores[i] = self.calculate_fitness_func(matrix, constraints_list, immutable_positions, self.max_score)
            scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
            bias_selection_array = []
            for key in scores.keys():
                for j in range(scores[key]):
                    bias_selection_array.append(key)

            avg_fit_score = self.calculate_avg_fitness_score(scores)
            # print score every 10 generations
            if generations % 10 == 0:
                print(f"generation: {generations}")
                print(f"max score: {scores[list(scores)[0]]}")
                print(f"avg score: {avg_fit_score}")
                print(matrices[list(scores)[0]])
                print()

            # check if we found the best solution with no mistakes in its fitness score
            if scores[list(scores)[0]] == self.max_score:
                print("found!")
                print(f"number of generations: {generations}")
                return matrices[list(scores)[0]]

            else:
                new_matrices = []  # matrix that will contain the new generation

                # in order to avoid premature convergence, every 200 generations we create new half of the initial
                #  number of boards and set the mutations percentage to be 50
                if (generations % 200 < 10 and generations > 10):
                    new, constraints_list, immutable_positions = self.generate_boards(self.num_of_boards // 2)
                    for m in new:
                        new_matrices.append(m)
                    mutations_percentage = 50

                # in order to avoid premature convergence,we check if the difference between the
                # best fitness score to the average fitness score is less than 1.
                # if so, we create more mutations
                elif abs(avg_fit_score - scores[list(scores)[0]]) < 1:
                    mutations_percentage = 30


                else:
                    mutations_percentage = 10
                num_mutations = mutations_percentage * len(matrices) // 100

                elitisim_list = []
                # replicate of elitism
                j = 0
                for k in scores.keys():
                    if j == num_elites:
                        break
                    new_matrices.append(matrices[k])
                    elitisim_list.append(k)
                    j += 1

                # create the other offsprings by crossover
                while len(new_matrices) < self.num_of_boards:
                    # random 2 parents
                    parent1 = random.choice(bias_selection_array)
                    parent2 = random.choice(bias_selection_array)
                    offspring1, offspring2 = self.crossover(matrices[parent1], matrices[parent2])
                    new_matrices.append(offspring1)
                    new_matrices.append(offspring2)

                new_matrices = np.array(new_matrices)

                # make mutations
                for i in range(num_mutations):
                    # random matrix number
                    mat_num = random.choice(bias_selection_array)

                    # we don't make a mutation to the elite solutions
                    if mat_num in elitisim_list:
                        continue
                    for p in range(self.board_size * 2):
                        # choosing to swap cols or rows
                        choice = random.randint(0, 1)
                        col_or_row1 = random.randint(0, self.board_size - 1)
                        col_or_row2 = random.randint(0, self.board_size - 1)
                        # random different rows/cols to swap
                        while col_or_row1 == col_or_row2:
                            col_or_row1 = random.randint(0, self.board_size - 1)
                            col_or_row2 = random.randint(0, self.board_size - 1)
                        if choice == 0:  # swap cols
                            tmp = new_matrices[mat_num][:, col_or_row1].copy()
                            new_matrices[mat_num][:, col_or_row1] = new_matrices[mat_num][:, col_or_row2].copy()
                            new_matrices[mat_num][:, col_or_row2] = tmp
                        else:  # swap rows
                            tmp = new_matrices[mat_num][col_or_row1].copy()
                            new_matrices[mat_num][col_or_row1] = new_matrices[mat_num][col_or_row2].copy()
                            new_matrices[mat_num][col_or_row2] = tmp

                        if self.calculate_fitness_func(new_matrices[mat_num], constraints_list, immutable_positions,
                                                       self.max_score) == self.max_score:
                            print("found!")
                            print(f"number of generations: {generations}")
                            return new_matrices[mat_num]

                matrices = new_matrices.copy()
                generations += 1

        print(f"the best solution found after {generations} generations is with score: {scores[list(scores)[0]]}")
        return matrices[list(scores)[0]]
