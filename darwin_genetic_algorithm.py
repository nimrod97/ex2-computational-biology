import time
from genetic_algorithm import GeneticAlgorithm
import numpy as np
import random


class DarwinGeneticAlgorithm(GeneticAlgorithm):
    # count mistakes that appear in rows of the matrix
    def row_mistakes(self, matrix, constraints_list):
        row_mistakes = 0
        for c in constraints_list:
            i1 = c.tup1[0]
            j1 = c.tup1[1]
            i2 = c.tup2[0]
            j2 = c.tup2[1]
            if matrix[i1][j1] <= matrix[i2][j2]:
                if i1 == i2:
                    row_mistakes += 1
        return row_mistakes

    def col_mistakes(self, matrix, constraints_list):
        # count mistakes that appear in columns of the matrix
        col_mistakes = 0
        for c in constraints_list:
            i1 = c.tup1[0]
            j1 = c.tup1[1]
            i2 = c.tup2[0]
            j2 = c.tup2[1]
            if matrix[i1][j1] <= matrix[i2][j2]:
                if j1 == j2:
                    col_mistakes += 1
        return col_mistakes

    def given_number_positions_mistakes(self, matrix, immutable_positions):
        # check if the numbers in the initial board are in the desired positions
        given_number_mistakes = 0
        for im in immutable_positions.keys():
            i = im[0]
            j = im[1]
            if matrix[i][j] != immutable_positions[im]:
                given_number_mistakes += 2
        return given_number_mistakes

    def permutate_mistakes(self, matrix):
        # check if each number appears once in each row and col
        permutate_mistakes = 0
        appearances_rows = dict()
        appearances_cols = dict()
        # validate that each number appears once in every row and column in the matrix
        for i in range(len(matrix)):
            appearances_rows.clear()
            appearances_cols.clear()
            for j in range(len(matrix)):
                if not matrix[i][j] in appearances_rows:
                    appearances_rows[matrix[i][j]] = 1
                else:
                    appearances_rows[matrix[i][j]] += 1
                if not matrix[j][i] in appearances_cols:
                    appearances_cols[matrix[j][i]] = 1
                else:
                    appearances_cols[matrix[j][i]] += 1

            for key in appearances_rows.keys():
                if appearances_rows[key] > 1:
                    permutate_mistakes += appearances_rows[key] - 1
            for key in appearances_cols.keys():
                if appearances_cols[key] > 1:
                    permutate_mistakes += appearances_cols[key] - 1
        return permutate_mistakes

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
            # update dictionary with new mistakes
            mistakes['row_mistakes'] = row_mistakes
            mistakes['col_mistakes'] = col_mistakes
            mistakes['given_number_mistakes'] = given_number_mistakes
            mistakes['permutate_mistakes'] = permutate_mistakes
            mistakes = dict(sorted(mistakes.items(), key=lambda item: item[1], reverse=True))
            # we handle the type of mistake that has the most number of mistakes
            to_handle = list(mistakes)[0]
            if mistakes[list(mistakes)[0]] == 0:
                return 0
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
                    # we try to swap 2 numbers in the row
                    if temp[i][j] != immutable_positions[im]:
                        col = random.randint(0, self.board_size - 1)
                        while col == j:
                            col = random.randint(0, self.board_size - 1)
                        t = temp[i][j]
                        temp[i][j] = temp[i][col]
                        temp[i][col] = t
            else:  # permutate mistakes - there exist rows or columns that not every number appears once
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

        # compare between the change in the total mistakes after the optimization trial
        # if there are less mistakes than before the optimization, we return the optimized number of mistakes
        # else, we return the number of mistakes as it was before the optimization

        total_now = score - (row_mistakes + col_mistakes + given_number_mistakes + permutate_mistakes)
        return total_now if total_now > initial else initial

    def solve(self, matrices, constraints_list, immutable_positions):
        elite_percentage = 5
        num_elites = elite_percentage * len(matrices) // 100
        generations = 0
        self.max_score = len(constraints_list) + len(immutable_positions) + 2 * (self.board_size) * (
                self.board_size - 1)
        optimized_scores = dict()
        scores = dict()
        print(f"desired score: {self.max_score}")
        print()
        time.sleep(1)
        while generations < 20000:
            # we prefer to let the better creatures the option to create the next generation
            # and this is the meaning of optimized scores - according to them, we rank the
            # creatures of this generation but we don't change them according to the optimized result
            for i, matrix in enumerate(matrices):
                optimized_scores[i] = self.optimize(matrix, constraints_list, immutable_positions)
                scores[i] = self.calculate_fitness_func(matrix, constraints_list, immutable_positions, self.max_score)
            scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
            optimized_scores = dict(sorted(optimized_scores.items(), key=lambda item: item[1], reverse=True))
            bias_selection_array = []
            for key in optimized_scores.keys():
                for j in range(optimized_scores[key]):
                    bias_selection_array.append(key)
            avg_opt_fit_score = self.calculate_avg_fitness_score(optimized_scores)
            # print score every 10 generations
            if generations % 10 == 0:
                print(f"generation: {generations}")
                print(f"max score: {scores[list(scores)[0]]}")
                print(f"avg score: {avg_opt_fit_score}")
                print(matrices[list(scores)[0]])
                print()
            # check if we found the best solution with no scores in its fitness score
            if scores[list(scores)[0]] == self.max_score:
                print("found!")
                print(f"number of generations: {generations}")
                return matrices[list(scores)[0]]
            else:
                new_matrices = []  # matrix that will contain the new generation
                if (generations % 200 < 10 and generations > 10):
                    new, constraints_list, immutable_positions = self.generate_boards(self.num_of_boards // 2)
                    for m in new:
                        new_matrices.append(m)
                    mutations_percentage = 50
                elif abs(avg_opt_fit_score - scores[list(scores)[0]]) < 1:
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

        print(f"the best solution found after {generations} generations is with score: {scores[list(scores)[0]]}")
        return matrices[list(scores)[0]]
