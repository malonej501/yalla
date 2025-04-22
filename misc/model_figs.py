import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

PLOT = 1  # 0: single force, 1: force for all cells
CELL_TYPE = 1  # 1: spot, 2: non-spot
EXPORT = True  # save plot as pdf
V = False  # verbose


def get_params(ctype):
    """
    Get params from file. Returns the parameters for the force equation.
    """
    pms = pd.read_csv("../params/default.csv")

    r_max = float(pms.loc[pms["param"] == "r_max", "value"].values[0])
    A, a, R, r = None, None, None, None
    if ctype == 1:
        A = float(pms.loc[pms["param"] == "Aii", "value"].values[0])
        a = float(pms.loc[pms["param"] == "aii", "value"].values[0])
        R = float(pms.loc[pms["param"] == "Rii", "value"].values[0])
        r = float(pms.loc[pms["param"] == "rii", "value"].values[0])
    elif ctype == 2:
        A = float(pms.loc[pms["param"] == "Add", "value"].values[0])
        a = float(pms.loc[pms["param"] == "add", "value"].values[0])
        R = float(pms.loc[pms["param"] == "Rdd", "value"].values[0])
        r = float(pms.loc[pms["param"] == "rdd", "value"].values[0])
    if V:
        print(f"cell type: {ctype}")
        print(f"r_max: {r_max}")
        print(f"A: {A}, a: {a}, R: {R}, r: {r}")

    return r_max, A, a, R, r


def force_equation():
    """
    Returns the force equation in a symbolic form.
    """

    # Define symbols
    A, a, R, r, d = sp.symbols('A a R r d')

    # term1 = A / a * sp.exp(-d / a)
    # term2 = R / r * sp.exp(-d / r)
    term1 = R * sp.exp(-d / r)  # from Volkening 2015 supp inf.
    term2 = A * sp.exp(-d / a)

    f_pot = term1 - term2  # potential force
    f = sp.diff(f_pot, d)  # force

    if V:
        print("Force potential:", f_pot)
        print("Force:", f)
    return f


def plot_force_equation():
    """
    Plots the force equation.
    """

    # Define parameters
    A, a, R, r, d = sp.symbols('A a R r d')

    r_max_val, A_val, a_val, R_val, r_val = get_params(CELL_TYPE)

    d_vals = np.linspace(0, r_max_val, 100)  # range for d

    # Calculate force values
    f_vals = [force_equation().subs({
        A: A_val,
        a: a_val,
        R: R_val,
        r: r_val,
        d: d_val}) for d_val in d_vals]

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.plot(d_vals, f_vals)
    plt.ylim(-0.05, 0.005)
    plt.xlabel("d")
    plt.ylabel('F')
    plt.grid(alpha=0.3)
    if CELL_TYPE == 1:
        plt.title('Force equation for spot cell')
    elif CELL_TYPE == 2:
        plt.title('Force equation for non-spot cell')

    plt.tight_layout()
    if EXPORT:
        plt.savefig('force_equation.pdf', bbox_inches='tight')
    plt.show()


def plot_force_equation_all_cells():
    """
    Plots the force equation for all cells.
    """

    # Define parameters
    A, a, R, r, d = sp.symbols('A a R r d')
    r_max_val, A_val, a_val, R_val, r_val = get_params(CELL_TYPE)
    d_vals = np.linspace(0, r_max_val, 100)  # range for d

    # Plotting
    fig, axs = plt.subplots(figsize=(5, 3), ncols=2, nrows=1, sharey=True)
    for i, ax in enumerate(axs):
        ctype = i + 1
        r_max_val, A_val, a_val, R_val, r_val = get_params(ctype)
        f_vals = [force_equation().subs({
            A: A_val,
            a: a_val,
            R: R_val,
            r: r_val,
            d: d_val}) for d_val in d_vals]

        ax.plot(d_vals, f_vals)
        ax.set_ylim(-0.02, 0.005)
        ax.grid(alpha=0.3)
        ax.annotate(f"A = {A_val}\na = {a_val}\nR = {r_val}"
                    f"\nr = {r_val}\nr_max = {r_max_val}",
                    xy=(1, 0), xycoords='axes fraction',
                    xytext=(-10, 10), textcoords='offset pixels',
                    ha='right', va='bottom',)
        if ctype == 1:
            ax.set_title('Spot')
        elif ctype == 2:
            ax.set_title('Non-spot')

    fig.supxlabel(r"Separation distance ($s$)")
    fig.supylabel(r"Force magnitude ($F_{i,j}$)")
    # fig.suptitle("Force function for all cell types")
    plt.tight_layout()
    if EXPORT:
        plt.savefig('force_equation_all_cells.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # force_equation()
    if PLOT == 0:
        plot_force_equation()
    elif PLOT == 1:
        plot_force_equation_all_cells()
