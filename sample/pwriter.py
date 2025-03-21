import pandas as pd

def params_to_header(params: pd.DataFrame):
    """Write parameters to a header file.

    Args:
        params (pd.DataFrame): DataFrame containing parameters.

    Returns:
        None
    """

    with open("../params/params.h", "w") as f:
        f.write("#ifndef PARAMS_H\n#define PARAMS_H\n\n")

        curr_group = None
        for _, row in params.iterrows():
            param = row["param"]
            type_ = row["type"]
            value = row["value"]
            group = row["group"]
            if group != curr_group: # split params into groups
                f.write(f"\n// {group}\n")
                curr_group = group
            f.write(f"const {type_} {param} = {value};\n")

        f.write("\n#endif")

if __name__ == "__main__":
    params = pd.read_csv("../params/default.csv")
    params_to_header(params)