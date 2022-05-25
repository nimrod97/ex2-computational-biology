from darwin_genetic_algorithm import DarwinGeneticAlgorithm
import numpy as np
import random
import time


class LemarkGeneticAlgorithm(DarwinGeneticAlgorithm):
    def optimize(self, matrix, constraints_list, immutable_positions):
        # for every matrix we make up to the number of board size optimizations
        score = self.max_score
        mistakes = dict()
        row_mistakes = self.row_mistakes(matrix, constraints_list)
        col_mistakes = self.col_mistakes(matrix, constraints_list)
        given_number_mistakes = self.given_number_positions_mistakes(matrix, immutable_positions)
        permutate_mistakes = self.permutate_mistakes(matrix)
        initial = score - (row_mistakes + col_mistakes + given_number_mistakes + permutate_mistakes)
        temp = matrix.copy()
        for k in range(self.board_size):
            mistakes['row_mistakes'] = row_mistakes
            mistakes['col_mistakes'] = col_mistakes
            mistakes['given_number_mistakes'] = given_number_mistakes
            mistakes['permutate_mistakes'] = permutate_mistakes
            mistakes = dict(sorted(mistakes.items(), key=lambda item: item[1], reverse=True))
            to_handle = list(mistakes)[0]
            if mistakes[list(mistakes)[0]] == 0:
                break
            if to_handle == 'row_mistakes':
                # swap cols
                col1 = random.randint(0, self.board_size - 1)
                col2 = random.randint(0, self.board_size - 1)
                while col1 == col2:
                    col1 = random.randint(0, self.board_size - 1)
                    col2 = random.randint(0, self.board_size - 1)
                t = temp[:, col1].copy()
                temp[:, col1] = temp[:, col2].copy()
                temp[:, col2] = t
            elif to_handle == 'col_mistakes':
                # swap rows
                row1 = random.randint(0, self.board_size - 1)
                row2 = random.randint(0, self.board_size - 1)
                while row1 == row2:
                    row1 = random.randint(0, self.board_size - 1)
                    row2 = random.randint(0, self.board_size - 1)
                t = temp[row1].copy()
                temp[row1] = temp[row2].copy()
                temp[row2] = t
            elif to_handle == 'given_number_mistakes':
                for im in immutable_positions.keys():
                    i = im[0]
                    j = im[1]
                    if temp[i][j] != immutable_positions[im]:
                        col = random.randint(0, self.board_size - 1)
                        while col == j:
                            col = random.randint(0, self.board_size - 1)
                        t = temp[i][j]
                        temp[i][j] = temp[i][col]
                        temp[i][col] = t
            else:  # permutate mistakes
                appearances_rows = dict()
                appearances_cols = dict()
                for i in range(len(temp)):
                    appearances_rows.clear()
                    appearances_cols.clear()
                    for p in range(1, self.board_size + 1):
                        appearances_cols[p] = 1
                        appearances_rows[p] = 1
                    for j in range(len(temp)):
                        appearances_rows[temp[i][j]] -= 1
                        appearances_cols[temp[j][i]] -= 1
                        if appearances_rows[temp[i][j]] == -1:
                            index_i, index_j = i, j
                        if appearances_cols[temp[j][i]] == -1:
                            index_i, index_j = j, i
                    for key in appearances_rows.keys():
                        if appearances_rows[key] == 1:
                            temp[index_i][index_j] = key
                            break
                    for key in appearances_cols.keys():
                        if appearances_cols[key] == 1:
                            temp[index_i][index_j] = key
                            break
            row_mistakes = self.row_mistakes(temp, constraints_list)
            col_mistakes = self.col_mistakes(temp, constraints_list)
            given_number_mistakes = self.given_number_positions_mistakes(temp, immutable_positions)
            permutate_mistakes = self.permutate_mistakes(temp)

        total_now = score - (row_mistakes + col_mistakes + given_number_mistakes + permutate_mistakes)
        # compare between the change in the total mistakes after the optimization trial
        # if there are less mistakes than before the optimization, we return the optimized matrix
        # else, we return the matrix before the optimization
        if total_now > initial:
            return total_now, temp
        # we encourage to vary the population
        elif total_now == initial:
            if not np.array_equal(temp, matrix):
                return total_now, temp
        return initial, matrix

    def solve(self, matrices, constraints_list, immutable_positions):
        elite_percentage = 5
        num_elites = elite_percentage * len(matrices) // 100
        generations = 0
        self.max_score = len(constraints_list) + len(immutable_positions) + 2 * (self.board_size) * (
                self.board_size - 1)
        optimized_scores = dict()
        print(f"desired score: {self.max_score}")
        print()
        time.sleep(1)
        while generations < 20000:
            for i, matrix in enumerate(matrices):
                score, matrix = self.optimize(matrix, constraints_list, immutable_positions)
                optimized_scores[i] = score
                matrices[i] = matrix
            optimized_scores = dict(sorted(optimized_scores.items(), key=lambda item: item[1], reverse=True))
            bias_selection_array = []
            for key in optimized_scores.keys():
                for j in range(optimized_scores[key]):
                    bias_selection_array.append(key)
            avg_opt_fit_score = self.calculate_avg_fitness_score(optimized_scores)
            # print score every 10 generations
            if generations % 10 == 0:
                print(f"generation: {generations}")
                print(f"max score: {optimized_scores[list(optimized_scores)[0]]}")
                print(f"avg score: {avg_opt_fit_score}")
                print(matrices[list(optimized_scores)[0]])
                print()

            # check if we found the best solution with no mistakes in its fitness score
            if optimized_scores[list(optimized_scores)[0]] == self.max_score:
                print("found!")
                print(f"number of generations: {generations}")
                return matrices[list(optimized_scores)[0]]
            else:
                new_matrices = []  # matrix that will contain the new generation

                # in order to avoid premature convergence, every 200 generations we create new half of the initial
                #  number of boards and set the mutations percentage to be 50
                if (generations % 200 < 10 and generations > 10):
                    new, constraints_list, immutable_positions = self.generate_boards(self.num_of_boards // 2)
                    for m in new:
                        new_matrices.append(m)
                    mutations_percentage = 50
                elif abs(avg_opt_fit_score - optimized_scores[list(optimized_scores)[0]]) < 1:
                    mutations_percentage = 30

                else:
                    mutations_percentage = 10
                num_mutations = mutations_percentage * len(matrices) // 100

                elitisim_list = []
                # replicate of elitism
                j = 0
                for k in optimized_scores.keys():
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
                    offspring1_score = self.calculate_fitness_func(offspring1, constraints_list, immutable_positions,
                                                                   self.max_score)
                    offspring2_score = self.calculate_fitness_func(offspring2, constraints_list, immutable_positions,
                                                                   self.max_score)
                    # as a part of optimization, we add to the new generation only offsprings that have better fitness
                    # score than at least one of its parents
                    if offspring1_score > optimized_scores[parent1] or offspring1_score > optimized_scores[
                        parent2]:
                        new_matrices.append(offspring1)
                    if offspring2_score > optimized_scores[parent1] or offspring2_score > optimized_scores[
                        parent2]:
                        new_matrices.append(offspring2)

                if len(new_matrices) > self.num_of_boards:
                    while len(new_matrices) != self.num_of_boards:
                        new_matrices.pop()
                new_matrices = np.array(new_matrices)

                # make mutations
                for i in range(num_mutations):
                    # random matrix number
                    mat_num = random.choice(bias_selection_array)

                    # we don't make a mutation to the elite solutions
                    if mat_num in elitisim_list:
                        continue
                    for p in range(self.board_size * 2):
                        choice = random.randint(0, 1)
                        # random different rows/cols to swap
                        col_or_row1 = random.randint(0, self.board_size - 1)
                        col_or_row2 = random.randint(0, self.board_size - 1)
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

        print(
            f"the best solution found after {generations} generations is with score: {optimized_scores[list(optimized_scores)[0]]}")
        return matrices[list(optimized_scores)[0]]
