import sys
import pandas as pd

PARAMS_PATH = "../params/default.csv"


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
            value = row["value"]
            group = row["group"]
            if group != curr_group:  # split params into groups
                f.write(f"\n\t// {group}\n")
                curr_group = group
            f.write(f"\t{type_} {param} = {value};\n")

        f.write("};\n\n")
        # v_list = params["value"].astype(str).tolist()
        # v_str = "{" + ", ".join(v_list) + "}"
        # f.write(f"Pm h_pm = {v_str};\n")
        f.write("Pm h_pm;\n")
        f.write("#endif")


def write_logging_instructions(params: pd.DataFrame):
    """Write C++ code to log parameters for each step of spot_finder.

    Args:
        params (pd.DataFrame): DataFrame containing parameters.

    Returns:
        None
    """
    # ensure param names have no surrounding whitespace
    param_list = params["param"].astype(str).str.strip().tolist()

    with open("../sample/logging.cpp", "w") as f:
        f.write("#include <fstream>\n")
        f.write("#include <string>\n")
        f.write('#include "../params/params.h"\n\n')

        # header: use commas without spaces
        f.write("void write_report_header(std::ofstream& file) {\n")
        step_info_h = "walk_id,step,attempt,status,target"
        header = step_info_h + "," + ",".join(param_list) if param_list else step_info_h
        f.write(f'\tfile << "{header}\\n";\n')
        f.write("}\n\n")

        # row: build the C++ expression piecewise so there are no spaces in output cells
        f.write("void write_report_row(std::ofstream& file, int walk_id, "
                "int i, int attempt, std::string status, std::string target, const "
                "Pm& h_pm) {\n")
        # base row (no trailing comma)
        f.write('\tfile << walk_id << "," << i << "," << attempt << "," << status << "," << target')
        # append each Pm field, prefixed by comma, no extra spaces
        for pname in param_list:
            f.write(f' << "," << h_pm.{pname}')
        f.write(' << "\\n";\n')
        f.write("}\n\n")


def print_help():
    """Prints help message for the script."""
    help_message = """
    Usage: python pwriter.py [options]

    Options:
        -h          Show this help message and exit
        -p          Parameter file to write from (default: default.csv)
    """
    print(help_message)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    else:
        if "-p" in args:
            PARAMS_PATH = "../params/" + str(args[args.index("-p") + 1])
        params = pd.read_csv(PARAMS_PATH)
        params_to_header(params)
        write_logging_instructions(params)
