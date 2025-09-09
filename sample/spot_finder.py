import pandas as pd
import numpy as np
from pwriter import params_to_header

targets = ["D_u","k_prod","k_deg","u_death","A_div"]

def mutate(p_curr, p_i):

    """Change the value of parameter p_i by a random amount"""

    muts = [
        -np.random.uniform(0, 0.1),
        -np.random.uniform(0.1, 1),
        -np.random.uniform(1, 10),
        np.random.uniform(0, 0.1),
        np.random.uniform(0.1, 1),
        np.random.uniform(1, 10)
    ]

    p_curr["value"][p_i] += np.random.choice(muts)


def test_params(p):

    params_to_header(p) # write parameters to header file

    # run simulation
    # get output
    # compare output to target
    # if output is close enough, stop
    # else, continue

def main():
    p_init = pd.read_csv("../params/default.csv")

    target_idxs = [int(p_init[p_init["param"] == t].index[0]) for t in targets]

    p_curr = p_init.copy()
    for step in range(0, 100):
        p_i = np.random.choice(target_idxs) # select target parameter index
        mutate(p_curr, p_i)
        test_params(p_curr)


if __name__ == "__main__":
    main()