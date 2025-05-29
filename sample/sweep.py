# Produce phenotypes for a range of values of selected parameters. Gives an
# overview of the model morphospace for select parameters.
import os
import pandas as pd
import subprocess
from scipy.stats import qmc
from pwriter import params_to_header

PARAMS = ["D_u", "k_prod", "k_deg", "u_death", "A_div"]
RANGES = {
    "D_u": (0.1, 1),  # min and max values for each parameter
    "k_prod": (0.1, 1),
    "k_deg": (0.1, 1),
    "u_death": (0.01, 1),
    "A_div": (0.001, 0.01)
}
STEP = 10  # no. steps between min and max for each parameter
N_SAMP = 100  # Number of LHS samples
N_FRAMES = 1  # no. simulation frames to generate per sample
RUN_ID = "test"
DEFAULT = pd.read_csv("../params/default.csv")


def gen_sweep_data():
    """Returns dataframe with all parameter values to be attempted."""

    data = {}
    for param in PARAMS:
        min_val, max_val = RANGES[param]
        step_size = (max_val - min_val) / (STEP - 1)
        values = [min_val + i * step_size for i in range(STEP)]
        data[param] = values

    combos = pd.MultiIndex.from_product(
        [data[param] for param in PARAMS], names=PARAMS)
    df = combos.to_frame(index=False)
    return df


def gen_lhs_data():
    """Returns dataframe with Latin Hypercube sampled parameter values."""

    sampler = qmc.LatinHypercube(d=len(PARAMS))
    sample = sampler.random(n=N_SAMP)  # shape: (N_SAMPLES, n_params)

    # Scale samples to parameter ranges
    l_bounds = [RANGES[p][0] for p in PARAMS]
    u_bounds = [RANGES[p][1] for p in PARAMS]
    scaled = qmc.scale(sample, l_bounds, u_bounds)

    df = pd.DataFrame(scaled, columns=PARAMS)
    return df


def export_to_log(d, i):
    """Saves the parameters to a log file."""

    log_path = f"../run/output/log_{RUN_ID}.csv"
    log_exists = os.path.exists(log_path)
    if i == 0 and log_exists:
        os.remove(log_path)

    with open(log_path, "a", encoding="utf-8") as f:
        if i == 0:
            header = ["sample"] + d["param"].values.tolist()
            f.write(",".join(header) + "\n")
        values = [str(i)] + [str(v) for v in d["value"].values.tolist()]
        f.write(",".join(values) + "\n")


def main():
    """Main function to generate and save parameter data."""

    run_dir = "../run"
    env = os.environ.copy()
    env["PATH"] = "/home/m/malone/packages/xvfb/usr/bin:" + env.get("PATH", "")
    subprocess.run(["rm", "-f", "exec"], check=True, cwd=run_dir)
    subprocess.run(["rm", "-rf", "output"], check=True, cwd=run_dir)

    df = gen_lhs_data()
    for i in range(N_SAMP):
        d = DEFAULT.copy()  # reset default params

        d.loc[d["param"] == "no_frames", "value"] = N_FRAMES
        for p in PARAMS:  # sub new set of param values
            d.loc[d["param"] == p, "value"] = df[p].values[i]
        params_to_header(d)

        subprocess.run(["nvcc", "-std=c++14", "-arch=sm_61",
                        "../examples/eggspot.cu", "-o", "exec"
                        ], check=True, cwd=run_dir)
        subprocess.run(["./exec", str(0), str(i)], check=True, cwd=run_dir)
        subprocess.run(["python3", "render.py", "output", "-s", str(i),
                        "-f", str(2)],
                       check=True, cwd=run_dir, env=env)
        export_to_log(d, i)


if __name__ == "__main__":
    main()
