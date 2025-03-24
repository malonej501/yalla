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
        f.write("struct Pm {\n")

        curr_group = None
        for _, row in params.iterrows():
            param = row["param"]
            type_ = row["type"]
            group = row["group"]
            if group != curr_group: # split params into groups
                f.write(f"\n\t// {group}\n")
                curr_group = group
            f.write(f"\t{type_} {param};\n")
        
        f.write("};\n\n")
        v_list = params["value"].astype(str).tolist()
        v_str = "{" + ", ".join(v_list) + "}"
        f.write(f"Pm h_pm = {v_str};\n")
        f.write("\n#endif")



if __name__ == "__main__":
    params = pd.read_csv("../params/default.csv")
    params_to_header(params)