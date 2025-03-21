import pandas as pd
from pwriter import params_to_header

def main():
    params = pd.read_csv("../params/default.csv")
    params_to_header(params)

if __name__ == "__main__":
    main()