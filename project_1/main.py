import os
from problem_1 import solve_problem_1
from problem_2 import solve_problem_2
from problem_3 import solve_problem_3
from problem_4 import solve_problem_4

if __name__ == '__main__':
    os.makedirs("output", exist_ok=True)

    print("\n--- Solving Problem 1 ---")
    solve_problem_1()

    print("\n--- Solving Problem 2 ---")
    solve_problem_2()

    print("\n--- Solving Problem 3 ---")
    solve_problem_3()

    print("\n--- Solving Problem 4 ---")
    solve_problem_4()

    print("\n✅ All problems solved. Check the 'output/' folder for results and plots.")
