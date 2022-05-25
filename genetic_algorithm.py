import numpy as np
import random
import constraint


class GeneticAlgorithm:

    def __init__(self, lst_settings,num_of_boards):
        self.lst_settings = lst_settings
        self.board_size = int(lst_settings[0])
        self.num_of_boards=num_of_boards

    def check_given_digits_location(self, offspring, immutable_positions):
        for im in immutable_positions.keys():
            i = im[0]
            j = im[1]
            if offspring[i][j] != immutable_positions[im]:
                return False
        return True

    def calculate_avg_fitness_score(self, mistakes):
        score = 0
        for k in mistakes.keys():
            score += mistakes[k]
        return score / len(mistakes)


    # we take each number in the first col of each input matrix and match the remaining numbers in the other matrix
    # that appear after that first number
    def crossover(self, m1, m2):
        offspring1 = m1.copy()
        offspring2 = m2.copy()
        for i in range(self.board_size):
            offspring1[i][0] = m1[i][0]
            offspring2[i][0] = m2[i][0]
            for j in range(self.board_size):
                if m2[j][0] == offspring1[i][0]:
                    offspring1[i][1:] = m2[j][1:]
                if m1[j][0] == offspring2[i][0]:
                    offspring2[i][1:] = m1[j][1:]
        return offspring1, offspring2


    def calculate_fitness_func(self, matrix, constraints_list, immutable_positions,score):
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
                    score-=1
            for key in appearances_cols.keys():
                if appearances_cols[key] > 1:
                    score-=1

        # check about the "<" constraints
        for c in constraints_list:
            i1 = c.tup1[0]
            j1 = c.tup1[1]
            i2 = c.tup2[0]
            j2 = c.tup2[1]
            if matrix[i1][j1] <= matrix[i2][j2]:
                score-=1

        # check if the given numbers in the initial state of the board changed positions in the matrix,
        # if so, the penalty will be higher
        for im in immutable_positions.keys():
            i = im[0]
            j = im[1]
            if matrix[i][j] != immutable_positions[im]:
                score-=2
        return score

    def generate_boards(self, amount):
        num_of_given_dig = int(self.lst_settings[1])
        immutable_positions = dict()
        matrix = np.zeros((self.board_size, self.board_size))
        # put in one matrix the given numbers in the initial state of the board
        for i in range(2, num_of_given_dig + 2):
            tmp = self.lst_settings[i].split(' ')
            matrix[int(tmp[0]) - 1][int(tmp[1]) - 1] = int(tmp[2])
            immutable_positions[((int(tmp[0]) - 1, int(tmp[1]) - 1))] = int(matrix[int(tmp[0]) - 1][int(tmp[1]) - 1])
        i += 1
        # update the constraints of "<"
        constraints_list = []
        for j in range(i + 1, len(self.lst_settings)):
            tmp = self.lst_settings[j].split(' ')
            constraints_list.append(
                constraint.Constraint((int(tmp[0]) - 1, int(tmp[1]) - 1), (int(tmp[2]) - 1, int(tmp[3]) - 1)))
        # generate given amount of matrices and random
        # rest of numbers in each matrix such that every number appears once in each row and column
        matrices = []
        while len(matrices) < amount:
            flag = False
            m = np.vstack(matrix)
            for j in range(self.board_size):
                for k in range(self.board_size):
                    if m[j][k] == 0:
                        num = random.randint(1, self.board_size)
                        c = 0
                        while num in m[j] or num in m[:, k] and c < 100:
                            num = random.randint(1, self.board_size)
                            c += 1
                        m[j][k] = num
                        if c >= 100:
                            flag = True
                            break
                if flag:
                    break
            if not flag:
                matrices.append(m)
        matrices = np.array(matrices)
        return matrices, constraints_list, immutable_positions


