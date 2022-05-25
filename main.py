from regular_genetic_algorithm import RegularGeneticAlgorithm
from darwin_genetic_algorithm import DarwinGeneticAlgorithm
from lemark_genetic_algorithm import LemarkGeneticAlgorithm
import tkinter as tk
from tkinter import filedialog

if __name__ == '__main__':
    print("choose input file with the correct format")
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename()
    board_settings = open(file, 'r')
    lst_settings = []
    for line in board_settings:
        lst_settings.append(line)

    print("press 0 for run regular genetic algorithm, 1 for run darwin genetic algorithm and 2 for"
          " run lemark genetic algorithm")
    x = input()
    x = int(x)
    num_of_boards = 400
    if x == 0:
        genetic_algorithm = RegularGeneticAlgorithm(lst_settings, num_of_boards)
        print("running regular genetic algorithm")
    elif x == 1:
        genetic_algorithm = DarwinGeneticAlgorithm(lst_settings, num_of_boards)
        print("running darwin genetic algorithm")

    else:
        genetic_algorithm = LemarkGeneticAlgorithm(lst_settings, num_of_boards)
        print("running lemark genetic algorithm")

    matrices, constraints_list, immutable_positions = genetic_algorithm.generate_boards(num_of_boards)
    sol = genetic_algorithm.solve(matrices, constraints_list, immutable_positions)
    print(sol)
