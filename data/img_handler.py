"""For handling and retrieving data on real fin images."""

import os
import pandas as pd


def load_imgs_from_directory(dir_path):
    """Load all image file paths from the specified directory.

    Args:
        dir_path (str): The path to the directory containing images.

    Returns:
        list: A list of file paths to images in the directory.
    """
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    img_paths = []

    print(f"Loading images from directory: {dir_path}")
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                img_paths.append(os.path.join(root, file))

    return img_paths


def count_taxa(img_paths):
    """Count the number of unique taxa based on image file names.

    Args:
        img_paths (list): A list of image file paths.

    Returns:
        int: The number of unique species.
    """
    # Extract population identifiers from file paths

    df = pd.DataFrame(img_paths, columns=["full_path"])
    df["id"] = df["full_path"].apply(
        lambda x: os.path.basename(x).split(".")[0])
    df["group"] = df["full_path"].apply(lambda x: x.split("/")[1])
    # df["id"].str.split("_", expand=True)
    df["genus"] = df["id"].str.split("_", expand=True)[0]
    df["species"] = df["genus"] + "_" + df["id"].str.split("_", expand=True)[1]
    site_cols = df["id"].str.split("_", expand=True).iloc[:, 2:]
    df["site"] = site_cols.apply(lambda row: "_".join(row.dropna()), axis=1)

    print(df)

    n_genera = df["genus"].nunique()
    n_species = df["species"].nunique()
    n_populations = df["id"].nunique()

    print(f"Number of unique genera: {n_genera}")
    print(f"Number of unique species: {n_species}")
    print(f"Number of unique populations: {n_populations}")


if __name__ == "__main__":
    dir_path = "wild_data"

    img_paths = load_imgs_from_directory(dir_path)
    count_taxa(img_paths)
