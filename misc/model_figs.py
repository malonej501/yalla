import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

PLOT = 0  # 0: single force, 1: force for all cells, 2: forces and potentials
INT_TYPE = 1  # 0: attraction, 1: pure repulsion
EXPORT = True  # save plot as pdf
V = True  # verbose


def get_params(itype):
    """
    Get params from file. Returns the parameters for the force equation.
    """
    pms = pd.read_csv("../params/default.csv")

    r_max = float(pms.loc[pms["param"] == "r_max", "value"].values[0])
    A, a, R, r = None, None, None, None
    if itype == 0:
        A = float(pms.loc[pms["param"] == "Aii", "value"].values[0])
        a = float(pms.loc[pms["param"] == "aii", "value"].values[0])
        R = float(pms.loc[pms["param"] == "Rii", "value"].values[0])
        r = float(pms.loc[pms["param"] == "rii", "value"].values[0])
    elif itype == 1:
        A = float(pms.loc[pms["param"] == "Add", "value"].values[0])
        a = float(pms.loc[pms["param"] == "add", "value"].values[0])
        R = float(pms.loc[pms["param"] == "Rdd", "value"].values[0])
        r = float(pms.loc[pms["param"] == "rdd", "value"].values[0])
    if V:
        print(f"cell type: {itype}")
        print(f"r_max: {r_max}")
        print(f"A: {A}, a: {a}, R: {R}, r: {r}")

    return r_max, A, a, R, r


def force_equation():
    """Returns the force equation in a symbolic form."""

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


def force_potential():
    """Returns force potential in symbolic form."""

    # Define symbols
    A, a, R, r, d = sp.symbols('A a R r d')

    term1 = R * sp.exp(-d / r)  # from Volkening 2015 supp inf.
    term2 = A * sp.exp(-d / a)

    f_pot = term1 - term2  # potential force

    return f_pot


def plot_force_equation():
    """Plots the force equation."""

    # Define parameters
    A, a, R, r, d = sp.symbols('A a R r d')
    r_max_val, A_val, a_val, R_val, r_val = get_params(INT_TYPE)
    d_vals = np.linspace(0, r_max_val, 100)  # range for d
    ints = ["spot-spot", "default"]

    # Calculate force values
    f_vals = [force_equation().subs({
        A: A_val, a: a_val, R: R_val, r: r_val, d: d_val
    }) for d_val in d_vals]

    # Plotting
    fig, ax = plt.subplots(figsize=(4, 3))
    for itype in [0, 1]:
        r_max_val, A_val, a_val, R_val, r_val = get_params(itype)
        f_vals = [force_equation().subs({
            A: A_val, a: a_val, R: R_val, r: r_val, d: d_val
        }) for d_val in d_vals]
        plt.plot(d_vals, f_vals, label=ints[itype], color=f"C{itype}")
    plt.ylim(-0.05, 0.005)
    plt.legend(title="Interaction", loc="lower right")
    plt.xlabel("Separation distance ($mm$)")
    plt.ylabel("Force ($mm t^{-1}$)")
    plt.grid(alpha=0.3)
    plt.title("")
    # if INT_TYPE == 1:
    #     plt.title('Force equation for attraction/repulsion')
    # elif INT_TYPE == 2:
    #     plt.title('Force equation for pure repulsion')

    plt.tight_layout()
    if EXPORT:
        plt.savefig('force_equation.svg', bbox_inches='tight')
    plt.show()


def plot_force_equation_all_cells():
    """
    Plots the force equation for all cells.
    """

    # Define parameters
    A, a, R, r, d = sp.symbols('A a R r d')
    r_max_val, A_val, a_val, R_val, r_val = get_params(INT_TYPE)
    d_vals = np.linspace(0, r_max_val, 100)  # range for d

    # Plotting
    fig, axs = plt.subplots(figsize=(5, 3), ncols=2, nrows=1)  # , sharey=True)
    for itype, ax in enumerate(axs):
        r_max_val, A_val, a_val, R_val, r_val = get_params(itype)
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
        if itype == 1:
            ax.set_title('Attraction/repulsion')
        elif itype == 2:
            ax.set_title('Pure repulsion')

    fig.supxlabel(r"Separation distance s ($\mu m$)")
    fig.supylabel(r"Force magnitude ($F_{i,j}$)")
    # fig.suptitle("Force function for all cell types")
    plt.tight_layout()
    if EXPORT:
        plt.savefig('force_equation_all_interactions.pdf', bbox_inches='tight')
    plt.show()


def plot_force_and_potential():
    """Plots force equation and potential altogether for all interaction 
    types."""

    # Define parameters
    ints = ["spot-spot", "default"]
    A, a, R, r, d = sp.symbols('A a R r d')
    r_max_val, A_val, a_val, R_val, r_val = get_params(INT_TYPE)
    d_vals = np.linspace(0, r_max_val, 100)  # range for d

    # 1 panel for forces, one for force potentials
    fig, axs = plt.subplots(figsize=(7, 3), ncols=2, nrows=1,
                            layout="constrained")

    for itype in [0, 1]:
        r_max_val, A_val, a_val, R_val, r_val = get_params(itype)

        fp_vals = [force_potential().subs({
            A: A_val, a: a_val, R: R_val, r: r_val, d: d_val
        }) for d_val in d_vals]
        axs[0].plot(d_vals, fp_vals, color=f"C{itype}", label=ints[itype])
        axs[0].set_ylabel("Force potential")

        f_vals = [force_equation().subs({
            A: A_val, a: a_val, R: R_val, r: r_val, d: d_val
        }) for d_val in d_vals]
        axs[1].plot(d_vals, f_vals, color=f"C{itype}")
        axs[1].set_ylabel(r"Force ($mm t^{-1}$)")
        # axs[1].set_ylim(-10)

    for ax in axs.flat:
        ax.set_xlim(0, r_max_val)
        ax.grid(alpha=0.3)
        # ax.set_yscale("log")
    fig.supxlabel(r"Separation distance ($mm $)")
    fig.legend(loc="outside right", title="Interaction")
    if EXPORT:
        plt.savefig('force_equation_all_interactions.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # force_equation()
    if PLOT == 0:
        plot_force_equation()
    elif PLOT == 1:
        plot_force_equation_all_cells()
    elif PLOT == 2:
        plot_force_and_potential()
