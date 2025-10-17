"""For handling and retrieving data on real fin images."""

import os
import pandas as pd
import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from vedo import *
import datetime
import statsmodels.formula.api as smf
from PIL import Image

matplotlib.use("pgf")
plt.style.use("../misc/stylesheet.mplstyle")


START_FROM = 171  # index of image to start from when landmarking
WD = None  # working directory for analysis
FUNC = 0


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
                if "lmk" in file:
                    continue  # Skip already landmarked images
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


class Landmarker:
    """
    Class for landmarking fin images with mouse clicks.

    - Start with landmarking the scale bar. Left side first.
    - Next label the posterior proximal corner of the fin, then the anterior
    proximal corner.
    - Then proceed to landmark the ray tips, anti-clockwise from anterior
    to posterior.
    - Landmark type switches automatically to edge after first two scale bar
    points and to fin automatically after the first two edge points.

    Press 'f' for fin landmark (red), 'e' for edge landmark (green).
        's' for scalebar landmark (blue).
    Press 'r' to remove the last point.
    Press 'n' to move to the next image.
    Press 'q' to quit and save all landmarks to a CSV file.
    """

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.imgs = pd.DataFrame({"path": load_imgs_from_directory(dir_path)})
        self.imgs["id"] = self.imgs["path"].apply(
            lambda x: os.path.basename(x).split("_")[0])
        self.imgs["date"] = self.imgs["path"].apply(
            lambda x: x.split("_")[-2])
        self.imgs["idx"] = self.imgs["path"].apply(
            lambda x: int(x.split("_")[-1].split(".")[0]))
        self.imgs = self.imgs.sort_values(
            by=["id", "idx"]).reset_index(drop=True)
        self.landmarks = {}  # {img_path: [(x, y, type), ...]}
        self.current_points = []
        self.current_types = []
        self.current_img = None
        self.current_img_path = None
        self.current_type = "f"  # "f" for fin, "e" for edge

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events to record landmark points."""
        no_sb = 2  # number of scale bar points
        no_prox_edge = 11  # number of proximal edge points
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            self.current_types.append(self.current_type)
            self.redraw_image()
            # Switch to edge after no_sb scalebar points
            if len(self.current_types) >= no_sb and \
                    self.current_types[-no_sb:] == ["s"]*no_sb:
                self.current_type = "e"
                print("Automatically switched to edge landmark (green)")
            # Switch to fin after no_prox_edge edge points
            elif len(self.current_types) >= no_sb + no_prox_edge and \
                    self.current_types[-no_prox_edge:] == ["e"]*no_prox_edge:
                self.current_type = "f"
                print("Automatically switched to fin landmark (red)")

    def redraw_image(self):
        """Redraw the image with current points."""
        img_copy = self.current_img.copy()
        for i, (pt, typ) in enumerate(zip(self.current_points, self.current_types)):
            if typ == "f":
                color = (0, 0, 255)  # red for fin
            elif typ == "e":
                color = (0, 255, 0)  # green for edge (anterior/posterior)
            elif typ == "s":
                color = (255, 0, 0)  # blue for scalebar
            else:
                color = (0, 0, 0)    # fallback
            cv2.circle(img_copy, pt, 5, color, -1)
            cv2.putText(img_copy, f"{i+1}{typ}", (pt[0]+8, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow(self.current_img_path, img_copy)
        self.current_img = img_copy

    def run(self, start_idx=0):
        """Run the landmarking process."""
        max_width, max_height = 2000, 1500
        for img_path in self.imgs["path"][start_idx:]:
            self.current_img_path = img_path
            img_blank = cv2.imread(img_path)
            img = cv2.imread(img_path)
            out_img_path = os.path.splitext(img_path)[0] + "_lmk.png"
            self.current_points = []
            self.current_types = []
            self.current_type = "s"
            h, w = img.shape[:2]
            scale = min(max_width / w, max_height / h, 1.0)
            if scale < 1.0:
                img_blank = cv2.resize(img_blank, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            self.current_img = img
            cv2.namedWindow(img_path)
            cv2.resizeWindow(img_path, max_width, max_height)
            cv2.setMouseCallback(img_path, self.mouse_callback)
            print(f"Displaying: {img_path}")
            print("Press 'f' for fin landmark (red), 'e' for edge landmark (green)," +
                  " 's' for scale-bar landmark (blue).")
            while True:
                cv2.imshow(img_path, self.current_img)
                key = cv2.waitKey(20)
                if key == ord('n'):  # Next image
                    self.landmarks[img_path] = list(
                        zip(self.current_points, self.current_types))
                    cv2.imwrite(out_img_path, self.current_img)
                    self.save_landmarks(
                        f"{os.path.basename(self.dir_path)}_lmks.csv")
                    break
                elif key == ord('r'):  # Remove last point
                    if self.current_points:
                        self.current_points.pop()
                        self.current_types.pop()
                    self.current_img = img_blank.copy()
                    self.redraw_image()
                elif key == ord('f'):  # Switch to fin
                    self.current_type = "f"
                    print("Landmark type: fin (red)")
                elif key == ord('e'):  # Switch to edge
                    self.current_type = "e"
                    print("Landmark type: edge (green)")
                elif key == ord('s'):  # Switch to scalebar
                    self.current_type = "s"
                    print("Landmark type: scalebar (blue)")
                elif key == ord('q'):  # Quit
                    self.landmarks[img_path] = list(
                        zip(self.current_points, self.current_types))
                    cv2.imwrite(out_img_path, self.current_img)
                    cv2.destroyAllWindows()
                    self.save_landmarks(
                        f"{os.path.basename(self.dir_path)}_lmks.csv")
                    return
            cv2.destroyAllWindows()
        self.save_landmarks()

    def save_landmarks(self, out_path="landmarks.csv"):
        """Save all landmarks to a CSV file."""
        rows = []
        for img, pts_types in self.landmarks.items():
            for i, ((x, y), typ) in enumerate(pts_types):
                rows.append({"image": os.path.basename(img),
                            "landmark": i, "x": x, "y": y, "type": typ})
        df = pd.DataFrame(rows)
        df.insert(0, 'id', df['image'].apply(lambda x: x.split('_')[0]))
        df.insert(1, 'date', df['image'].apply(lambda x: x.split('_')[-2]))
        df.insert(2, 'idx', df['image'].apply(
            lambda x: int(x.split('_')[-1].split('.')[0])))
        df.to_csv(out_path, index=False)
        print(f"Landmarks saved to {out_path}")


def landmarks_to_vtk(path, transform=True, wallnodes=True, landmarks=True):
    """Convert landmark points to VTK format for 3D visualization."""

    lmks = pd.read_csv(path)

    print(lmks)
    for fish in lmks["id"].unique():
        f_lmks = lmks[lmks["id"] == fish]
        for i, date in enumerate(f_lmks["date"].unique()):

            f_lmks_i = f_lmks[f_lmks["date"] == date]
            vtk_path = os.path.join(
                os.path.dirname(path), f"{fish}_{date}_{i}_lmk.vtk")

            # Translate and rotate landmarks to standard orientation
            if transform:
                f_lmks_i = transform_landmarks(f_lmks_i)

            # Convert from pixels to mm (assuming 1mm scale bar)
            scale_bar = f_lmks_i[f_lmks_i["type"] == "s"]
            if len(scale_bar) != 2:
                print(
                    f"Skipping {fish}_{date}_{i}: need exactly 2 scale bar "
                    + f"points, found {len(scale_bar)}")
                continue
            sb_len = ((scale_bar.iloc[0]["x"] - scale_bar.iloc[1]["x"])**2 +
                      (scale_bar.iloc[0]["y"] - scale_bar.iloc[1]["y"])**2
                      )**0.5
            print(f"1mm scale bar len {fish}_{date}_{i}: {sb_len:.2f} pixels")

            # Divide x and y by scale bar length to get mm
            f_lmks_i = f_lmks_i.copy()  # Make an explicit copy
            f_lmks_i.loc[:, "x_mm"] = round(f_lmks_i["x"] / sb_len, 3)
            f_lmks_i.loc[:, "y_mm"] = round(f_lmks_i["y"] / sb_len, 3)

            # remove scale bar points
            f_lmks_i = f_lmks_i[f_lmks_i["type"] != "s"]

            # Calculate wall normals
            f_lmks_i = get_wallnorms(f_lmks_i)
            f_lmks_i = f_lmks_i.round(5)  # round to avoid precision issues

            with open(vtk_path, 'w', encoding='utf-8') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write(f"{fish} wallnodes\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")

                if wallnodes and landmarks:
                    f.write(f"POINTS {len(f_lmks_i)} float\n")
                    for _, row in f_lmks_i.iterrows():
                        f.write(f"{row['x_mm']} {row['y_mm']} 0.0\n")
                    f.write(f"POLYGONS 1 {len(f_lmks_i) + 1}\n")
                    f.write(f"{len(f_lmks_i)} ")
                    f.write(" ".join(str(j) for j in range(len(f_lmks_i))))
                    f.write(f"\nPOINT_DATA {len(f_lmks_i)}\n")
                    f.write("NORMALS polarity float\n")
                    for _, row in f_lmks_i.iterrows():
                        f.write(f"{row['x_pol']} {row['y_pol']} 0.0\n")

                if landmarks and not wallnodes:
                    f_lmks_i = f_lmks_i[f_lmks_i["type"] != "w_node"]
                    f.write(f"POINTS {len(f_lmks_i)} float\n")
                    for _, row in f_lmks_i.iterrows():
                        f.write(f"{row['x_mm']} {row['y_mm']} 0.0\n")
                    f.write(f"POLYGONS 1 {len(f_lmks_i) + 1}\n")
                    f.write(f"{len(f_lmks_i)} ")
                    f.write(" ".join(str(j) for j in range(len(f_lmks_i))))

                if wallnodes and not landmarks:
                    f_lmks_i = f_lmks_i[f_lmks_i["type"] == "w_node"]
                    print(f_lmks_i)
                    print(f_lmks_i[["x_mm", "y_mm"]].values.tolist())
                    print(list(range(len(f_lmks_i))))
                    # v = [(row["x_mm"], row["y_mm"], 0)
                    #      for _, row in f_lmks_i.iterrows()]
                    # faces = [list(range(len(v)))]
                    # mesh = Mesh([v, faces])
                    # mesh.triangulate()  # triangulate for yalla compatability
                    f.write(f"POINTS {len(f_lmks_i)} float\n")
                    for _, row in f_lmks_i.iterrows():
                        f.write(f"{row['x_mm']} {row['y_mm']} 0.0\n")
                    # f.write(f"VERTICES {len(f_lmks_i)} {len(f_lmks_i)*2}\n")
                    # for j in range(len(f_lmks_i)):
                    #     f.write(f"1 {j}\n")
                    f.write(f"POINT_DATA {len(f_lmks_i)}\n")
                    f.write("NORMALS polarity float\n")
                    for _, row in f_lmks_i.iterrows():
                        f.write(f"{row['x_pol']} {row['y_pol']} 0.0\n")
                    # f.write(
                    #     f"POLYGONS {len(mesh.cells)} {len(mesh.cells) +
                    #       len(mesh.vertices)}\n")
                    # for face in mesh.cells:
                    #     f.write(f"{len(face)} ")
                    #     f.write(" ".join(str(v) for v in face))
                    #     f.write("\n")

            print(f"VTK saved to {vtk_path}")


def load_landmark_vtks(dir_path):
    """Load all VTK landmark files from the specified directory.

    Args:
        dir_path (str): The path to the directory containing VTK files.
    Returns:
        meshes (list): A list of Mesh objects loaded from VTK files in
        the directory.
    """
    vtk_files = [os.path.join(dir_path, f) for f in os.listdir(
        dir_path) if f.endswith('.vtk')]
    if not vtk_files:
        print(f"No VTK files found in directory: {dir_path}")
        return

    # Sort by the number after the last '_' and before the '.'
    # vtk_files = sorted(
    #     vtk_files,
    #     key=lambda x: int(os.path.basename(x).split('_')[-2])
    # )
    # group by fish id
    # fish_ids = sorted({os.path.basename(f).split('_')[0] for f in vtk_files},
    #                   key=lambda x: int(x.split('-')[-1]))
    # vtk_files_grouped = []
    # for fish in fish_ids:
    #     fish_files = [
    #         f for f in vtk_files if os.path.basename(f).startswith(fish)]
    #     fish_files = sorted(
    #         fish_files,
    #         key=lambda x: int(os.path.basename(x).split('_')[-2])
    #     )
    #     vtk_files_grouped.append(fish_files)
    vtk_files = pd.DataFrame({"path": vtk_files})
    vtk_files["fish_id"] = [os.path.basename(
        f).split("_")[0] for f in vtk_files["path"]]
    vtk_files["date"] = [os.path.basename(
        f).split("_")[1] for f in vtk_files["path"]]
    vtk_files["date"] = pd.to_datetime(vtk_files["date"], format="%d-%m")
    vtk_files["day"] = (vtk_files["date"] - vtk_files["date"].min()).dt.days
    vtk_files["stage"] = [int(os.path.basename(
        f).split("_")[2]) for f in vtk_files["path"]]
    vtk_files = vtk_files.sort_values(["fish_id", "stage"])
    vtk_files = vtk_files.reset_index(drop=True)

    # Remove known problematic file - the scale bar is wrong
    # vtk_files.remove("lmk_DA-1-10_12-09-25/DA-1-10_17-07_1_lmk.vtk")

    # meshes = [[Mesh(f) for f in group] for group in vtk_files_grouped]
    vtk_files["mesh"] = [Mesh(f).triangulate() for f in vtk_files["path"]]
    vtk_files["area"] = [m.area() for m in vtk_files["mesh"]]

    return vtk_files


def visualise_landmark_vtks(dir_path):
    """Use matplotlib to plot the mesh time series."""

    vtks = load_landmark_vtks(dir_path)
    # meshes_trans = transform_landmarks(vtk_files, t=True, r=True)
    suffix = ""

    # Determine global x and y limits
    pad = 0.3  # Add padding to limits
    all_x = []
    all_y = []
    for mesh in vtks["mesh"]:
        if mesh.area() > 0.05:    # likely there was no scale bar
            continue
        all_x.extend(mesh.vertices[:, 0])
        all_y.extend(mesh.vertices[:, 1])

    x_min, x_max = min(all_x) - pad, max(all_x) + pad
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    # fish_id is columns, stage is rows
    fig, axs = plt.subplots(figsize=(16, 16), nrows=24, ncols=10,
                            # sharex=True, sharey=True,
                            layout="constrained")

    for i, stage in enumerate(sorted(vtks["stage"].unique())):
        for j, fish_id in enumerate(sorted(vtks["fish_id"].unique())):
            print(i)
            print(i, j)
            ax = axs[stage][j]
            mesh = vtks[(vtks["fish_id"] == fish_id) &
                        (vtks["stage"] == stage)]
            print(mesh)
            # print(mesh)
            if len(mesh) == 0:
                print(f"No mesh for {fish_id} {stage}")
                continue
            mesh = mesh["mesh"].values[0]
            if mesh.area() > 0.05:    # likely there was no scale bar
                print(f"Skipping {fish_id} {stage}, area too large: "
                      + f"{mesh.area():.3f} mm^2")
                continue

            # if mesh contains wall nodes
            if "polarity" in mesh.pointdata.keys():
                pdata = pd.DataFrame(
                    {"x": mesh.vertices[:, 0],  # get plot data
                     "y": mesh.vertices[:, 1],
                     "polarity_x": mesh.pointdata["polarity"][:, 0],
                     "polarity_y": mesh.pointdata["polarity"][:, 1]})
                pdata["type"] = ((pdata["polarity_x"] == 0) &
                                 (pdata["polarity_y"] == 0)).map(
                    {True: "lmk", False: "wnode"})
                if i == 0 and j == 0:
                    suffix += "_wnodes"
            else:
                pdata = pd.DataFrame({"x": mesh.vertices[:, 0],
                                      "y": mesh.vertices[:, 1],
                                     "type": "lmk"})

            # Plot landmarks
            lmks = pdata[pdata["type"] == "lmk"]
            if not lmks.empty:
                ax.scatter(lmks["x"], lmks["y"], color='C0')
                x = lmks["x"].values
                y = lmks["y"].values
                # Connect all points in order, including last to first
                x_closed = list(x) + [x[0]]
                y_closed = list(y) + [y[0]]
                ax.plot(x_closed, y_closed, color='C0', linewidth=1)
                for i, row in lmks.iterrows():
                    ax.text(row["x"], row["y"], str(
                        i), fontsize=9, color='red')
                if i == 0 and j == 0:
                    suffix += "_lmks"

            if "polarity" in mesh.pointdata.keys():
                # Draw wall normal vectors
                w_pts = pdata[pdata["type"] == "wnode"]
                ax.scatter(w_pts["x"], w_pts["y"], color='black')
                ax.quiver(w_pts["x"], w_pts["y"], w_pts["polarity_x"],
                          # scale_units='xy',
                          w_pts["polarity_y"], color='black', angles='xy',
                          scale=5, width=0.003, alpha=0.7)

            ax.set_title(f"{os.path.basename(mesh.filename)}", fontsize=4)
            ax.grid(alpha=0.3)
            ax.set_aspect("equal")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.invert_yaxis()

    fig.supxlabel('X (mm)')
    fig.supylabel('Y (mm)')

    for ax in axs.flatten():
        if not ax.has_data():
            ax.axis("off")  # switch off unused subplots

    # plt.savefig(f"{dir_path}_{suffix}.png", dpi=600)
    plt.show()


def visualise_landmarks_layered(dir_path, text=False, norm=False):
    """Plot landmark timeseries but one axes per fish_id."""
    max_area = 0.1  # skip meshes with area above this (likely no scale bar)
    alpha = 0.7
    plot_landmarks = False

    vtks = load_landmark_vtks(dir_path)
    suffix = ""

    # Determine global x and y limits
    pad = 0.3  # Add padding to limits
    all_x = []
    all_y = []
    for mesh in vtks["mesh"]:
        if mesh.area() > max_area:    # likely there was no scale bar
            continue
        if mesh.vertices[:, 1].min() < -6:  # likely bad mesh
            continue
        all_x.extend(mesh.vertices[:, 0])
        all_y.extend(mesh.vertices[:, 1])

    x_min, x_max = min(all_x) - pad, max(all_x) + pad
    y_min, y_max = min(all_y) - pad, max(all_y) + pad

    # prepare colour map
    cnorm = plt.Normalize(vmin=0, vmax=vtks["day"].max())
    cmap = plt.cm.get_cmap("viridis")

    # fish_id is columns, stage is rows
    fig, axs = plt.subplots(figsize=(5, 5), nrows=5, ncols=2,
                            sharex=True, sharey=True,
                            layout="constrained")

    for i, stage in enumerate(sorted(vtks["stage"].unique())):
        for j, fish_id in enumerate(sorted(vtks["fish_id"].unique())):
            ax = axs.flatten()[j]
            mesh = vtks[(vtks["fish_id"] == fish_id) &
                        (vtks["stage"] == stage)]
            if len(mesh) == 0:
                print(f"No mesh for {fish_id} {stage}")
                continue
            color = cmap(cnorm(mesh["day"].values[0]))
            mesh = mesh["mesh"].values[0]

            if mesh.area() > max_area:    # likely there was no scale bar
                print(f"Skipping {fish_id} {stage}, area too large: "
                      + f"{mesh.area():.3f} mm^2")
                continue
            if mesh.vertices[:, 1].min() < -6:  # likely bad mesh
                print(f"Skipping {fish_id} {stage}, min y too small: "
                      + f"{mesh.vertices[:, 1].min():.3f} mm")
                continue

            # if mesh contains wall nodes
            if "polarity" in mesh.pointdata.keys():
                pdata = pd.DataFrame(
                    {"x": mesh.vertices[:, 0],  # get plot data
                     "y": mesh.vertices[:, 1],
                     "polarity_x": mesh.pointdata["polarity"][:, 0],
                     "polarity_y": mesh.pointdata["polarity"][:, 1]})
                pdata["type"] = ((pdata["polarity_x"] == 0) &
                                 (pdata["polarity_y"] == 0)).map(
                    {True: "lmk", False: "wnode"})
                if i == 0 and j == 0:
                    suffix += "_wnodes"
            else:
                pdata = pd.DataFrame({"x": mesh.vertices[:, 0],
                                      "y": mesh.vertices[:, 1],
                                     "type": "lmk"})

            # Plot landmarks
            lmks = pdata[pdata["type"] == "lmk"]
            if not lmks.empty:
                if plot_landmarks:
                    ax.scatter(lmks["x"], lmks["y"], color=color,
                               alpha=alpha)
                x = lmks["x"].values
                y = lmks["y"].values
                # Connect all points in order, including last to first
                x_closed = list(x) + [x[0]]
                y_closed = list(y) + [y[0]]
                ax.plot(x_closed, y_closed,
                        color=color, linewidth=1, alpha=alpha)
                if text:
                    for i, row in lmks.iterrows():
                        ax.text(row["x"], row["y"], str(
                            i), fontsize=9, color='red')
                if i == 0 and j == 0:
                    suffix += "_lmks"

            if ("polarity" in mesh.pointdata.keys()) and norm:
                # Draw wall normal vectors
                w_pts = pdata[pdata["type"] == "wnode"]
                ax.scatter(w_pts["x"], w_pts["y"], color='black')
                ax.quiver(w_pts["x"], w_pts["y"], w_pts["polarity_x"],
                          # scale_units='xy',
                          w_pts["polarity_y"], color='black', angles='xy',
                          scale=5, width=0.003, alpha=0.7)

            ax.set_title(f"{os.path.basename(mesh.filename)}", fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_aspect("equal")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # ax.set_face
            # color("gray")
            # ax.invert_yaxis()

    fig.supxlabel('Anterior-posterior (mm)')
    fig.supylabel('Proximal-distal (mm)')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])

    fig.colorbar(sm, ax=axs, orientation='vertical',
                 label="Days after first image", shrink=0.5)
    for ax in axs.flatten():
        if not ax.has_data():
            ax.axis("off")  # switch off unused subplots

    plt.savefig(f"{dir_path}_{suffix}.svg")
    plt.show()


def transform_landmarks(lmks, tl=True, rot=True, ref=True):
    """Transform landmark lmks to standard orientation and position.
    Applies a translation and rotation based on the first two edge
    landmarks (posterior and anterior proximal corners of the fin).

    Args:
        lmks (pd.DataFrame): DataFrame containing landmark data for
            a single fish image.
        tl (bool): Whether to apply translation.
        rot (bool): Whether to apply rotation.
        ref (bool): Whether to reflect in the x-axis.
    Returns:
        pd.DataFrame: Transformed landmark DataFrame.
    """

    # Get anterior and posterior edge points, assume order post, ant
    prox = lmks[lmks["type"] == "e"][["x", "y"]].values
    p, a = prox[0], prox[-1]  # posterior and anterior proixmal points

    if tl:
        lmks.loc[:, "x"] = lmks["x"] - a[0]
        lmks.loc[:, "y"] = lmks["y"] - a[1]

        prox = lmks[lmks["type"] == "e"][["x", "y"]].values
        p, a = prox[0], prox[-1]  # posterior and anterior proximal points

    if rot:
        ap = p - a  # vector from anterior to posterior edge point
        angle_rad = -(np.arctan2(ap[1], ap[0]))
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        def rotate_point(x, y):
            x_new = x * cos_angle - y * sin_angle
            y_new = x * sin_angle + y * cos_angle
            return x_new, y_new

        lmks[["x", "y"]] = lmks.apply(
            lambda row: rotate_point(row["x"], row["y"]),
            axis=1, result_type='expand')

    if ref:
        lmks.loc[:, "y"] = -lmks["y"]

    return lmks


def get_wallnorms(lmks, n=1):
    """Compute midpoint of mesh line segements and associated
    normal vector facing away from the mesh interior.
    Args:
        lmks (pd.DataFrame): DataFrame containing landmark data for
            a single fish image.
        n (int): Number of wall nodes to insert between each pair of
            landmarks. Default is 1.
    Returns:
        lmks_norm (pd.DataFrame): DataFrame with additional rows for
            wall midpoints and normal vectors.
    """
    wall_norms = []
    for i in range(len(lmks)):
        p1 = lmks.iloc[i][["x_mm", "y_mm"]].values
        if i == len(lmks) - 1:  # we wrap around to the first point
            p2 = lmks.iloc[0][["x_mm", "y_mm"]].values
        else:
            p2 = lmks.iloc[i + 1][["x_mm", "y_mm"]].values

        norm = np.array([(p2[1] - p1[1]), -1*(p2[0] - p1[0])])
        norm /= np.linalg.norm(norm)  # Normalize the normal vector
        # Generate n equally spaced points between p1 and p2 (excluding
        # endpoints)
        for k in range(1, n+1):
            frac = k / (n+1)
            node = p1 + frac * (p2 - p1)
            wall_norms.append({
                "x_mm": node[0],
                "y_mm": node[1],
                "x_pol": norm[0],
                "y_pol": norm[1],
                "type": "w_node"
            })
    wall_norms = pd.DataFrame(wall_norms)
    lmks_norm = pd.concat([lmks, wall_norms], axis=0, ignore_index=True)
    lmks_norm[["x_pol", "y_pol"]] = lmks_norm[["x_pol", "y_pol"]].fillna(0.0)

    return lmks_norm


def px_to_mm(f_lmks_i, transform=True,):
    """Convert landmark coordinates into standardised form in mm."""

    # f_lmks_i = f_lmks[f_lmks["date"] == date]
    fish = f_lmks_i["id"].iloc[0]
    date = f_lmks_i["date"].iloc[0]
    i = f_lmks_i["idx"].iloc[0]
    # print(date)

    # Translate and rotate landmarks to standard orientation
    if transform:
        f_lmks_i = transform_landmarks(f_lmks_i)

    # Convert from pixels to mm (assuming 1mm scale bar)
    scale_bar = f_lmks_i[f_lmks_i["type"] == "s"]
    if len(scale_bar) != 2:
        print(
            f"Skipping {fish}_{date}: need exactly 2 scale bar "
            + f"points, found {len(scale_bar)}")
        return
    sb_len = ((scale_bar.iloc[0]["x"] - scale_bar.iloc[1]["x"])**2 +
              (scale_bar.iloc[0]["y"] - scale_bar.iloc[1]["y"])**2
              )**0.5
    # print(f"1mm scale bar len {fish}_{date}_{i}: {sb_len:.2f} pixels")

    # Divide x and y by scale bar length to get mm
    f_lmks_i = f_lmks_i.copy()  # Make an explicit copy
    f_lmks_i.loc[:, "x_mm"] = round(f_lmks_i["x"] / sb_len, 3)
    f_lmks_i.loc[:, "y_mm"] = round(f_lmks_i["y"] / sb_len, 3)

    return f_lmks_i


def ap_len_over_time(path):
    """Plot length from proximal anterio to proximal posterior over time."""

    lmks = pd.read_csv(path)

    lmks = lmks.groupby(["id", "date"]).apply(
        px_to_mm).reset_index(drop=True)
    lmks["date"] = pd.to_datetime(lmks["date"], format="%d-%m")
    lmks["days"] = (lmks["date"] - lmks["date"].min()).dt.days

    max_lens = lmks.groupby(["id", "days"]).apply(
        lambda x: max(x[x["type"] == "e"]["x_mm"]) -
        min(x[x["type"] == "e"]["x_mm"])
    ).reset_index(name="ap_len_mm")

    # remove outliers
    max_lens = max_lens[max_lens["ap_len_mm"] < 10]
    print(max_lens)

    ids = sorted(max_lens["id"].unique(),
                 key=lambda x: int(x.split('-')[-1]))

    fig, ax = plt.subplots()

    for i in ids:
        filt = max_lens[max_lens["id"] == i]
        filt = filt[filt["ap_len_mm"] < 10]  # filter out bad data
        ax.plot(filt["days"], filt["ap_len_mm"], marker='o', label=i,
                alpha=0.3, markersize=3)

    ax.set_xlabel("Days since first image")
    ax.set_ylabel("AP length (mm)")
    ax.legend(title="Fish ID", fontsize=6)
    ax.grid(alpha=0.3)
    # see https://en.wikipedia.org/wiki/Mixed_model
    model = smf.mixedlm("ap_len_mm ~ days", data=max_lens,
                        groups=max_lens["id"],
                        # re_formula="~days", # for random slopes
                        ).fit()
    pred = model.predict(pd.DataFrame({"days": np.linspace(
        0, max_lens["days"].max(), max_lens["days"].max()+1)}))
    print(model.summary())
    ci = model.conf_int(alpha=0.05)
    print(ci)
    ci_int = ci.loc["Intercept"]
    ci_slope = ci.loc["days"]
    fit_u = (ci_int[1] + ci_slope[1] *
             np.arange(0, max_lens["days"].max()))
    fit_l = (ci_int[0] + ci_slope[0] *
             np.arange(0, max_lens["days"].max()))
    ax.plot(pred, color='black',
            linestyle='--', label='Linear regression')
    ax.fill_between(np.arange(0, max_lens["days"].max()), fit_l,
                    fit_u, color='gray',
                    alpha=0.5, label='95% Confidence Interval')
    text = (f"Slope: {model.params['days']:.3f} mm/day" + "\n" +
            f"Intercept: {model.params['Intercept']:.3f} mm")
    ax.text(0.95, 0.05, text, transform=ax.transAxes, va='bottom', ha="right",)
    plt.savefig("ap_len_over_time.pdf", bbox_inches='tight')
    plt.show()


def ray_len_over_time(path):
    """Plot length of each fin ray over time."""

    lmks = pd.read_csv(path)

    lmks = lmks.groupby(["id", "date"]).apply(
        px_to_mm).reset_index(drop=True)
    lmks["date"] = pd.to_datetime(lmks["date"], format="%d-%m")
    lmks["days"] = (lmks["date"] - lmks["date"].min()).dt.days
    lmks.sort_values(["id", "days", "landmark"], inplace=True)

    # get length of each ray (posterior distal pair)
    # lmks.groupby(["id", "days"]).apply(
    #     lambda x:
    # )
    ray_lens = []
    for image in lmks["image"].unique():
        print(image)
        f_lmks_i = lmks[lmks["image"] == image]

        # get edge and distal points in order from anterior to posterior
        edge_pts = f_lmks_i[f_lmks_i["type"] == "e"].sort_values(
            "landmark", ascending=False)
        dist_pts = f_lmks_i[f_lmks_i["type"] == "f"].sort_values(
            "landmark", ascending=True)

        if len(edge_pts) != len(dist_pts):  # check for any non-pairs
            print(
                f"Skipping {image}: number of edge points "
                + f"({len(edge_pts)}) does not match number of "
                + f"distal points ({len(dist_pts)})")
            continue

        ray_dists = np.linalg.norm(  # get distances between ray pairs in mm
            edge_pts[["x_mm", "y_mm"]].values -
            dist_pts[["x_mm", "y_mm"]].values, axis=1)
        ray_dat = pd.DataFrame({
            "id": f_lmks_i["id"].iloc[0],
            "date": f_lmks_i["date"].iloc[0],
            "days": f_lmks_i["days"].iloc[0],
            "idx": f_lmks_i["idx"].iloc[0],
            "ray_idx": range(0, len(ray_dists)),  # expected order a->p
            "ray_len_mm": ray_dists
        })
        ray_lens.append(ray_dat)
    ray_lens = pd.concat(ray_lens, axis=0, ignore_index=True)
    print(ray_lens)
    # remove outliers
    ray_lens = ray_lens[ray_lens["ray_len_mm"] < 10]

    # Plot ray length over time for each ray for each fish

    fig, axs = plt.subplots(figsize=(8, 4), nrows=2, ncols=6,
                            layout="constrained", sharex=True, sharey=True)

    reg_slopes = []
    labels, handles = [], []
    for i, ray_idx in enumerate(sorted(ray_lens["ray_idx"].unique())):
        ax = axs.flatten()[i]
        ray_i_lens = ray_lens[ray_lens["ray_idx"] == ray_idx]
        for fish_id in ray_i_lens["id"].unique():
            ray_lens_ij = ray_i_lens[ray_i_lens["id"] == fish_id]
            ax.plot(ray_lens_ij["days"], ray_lens_ij["ray_len_mm"],
                    marker='o', alpha=0.3, label=f"{fish_id}", markersize=2)
            if fish_id not in labels:
                labels.append(fish_id)
                handles.append(ax.lines[-1])  # get last line added to ax

        # fit LMM see https://en.wikipedia.org/wiki/Mixed_model
        model = smf.mixedlm("ray_len_mm ~ days", data=ray_i_lens,
                            groups=ray_i_lens["id"],
                            # re_formula="~days", # for random slopes
                            ).fit()
        pred = model.predict(pd.DataFrame({"days": np.linspace(
            0, ray_i_lens["days"].max(), ray_i_lens["days"].max()+1)}))

        print(pred)
        print(model.summary())
        reg_slopes.append(
            pd.DataFrame({"ray": ray_idx, "slope": model.params["days"],
                          "ci_lower": model.conf_int().loc["days"][0],
                          "ci_upper": model.conf_int().loc["days"][1],
                          "pval": model.pvalues["days"]}, index=[0]))

        ax.plot(pred, color='black',
                linestyle='--', label='Linear regression')

        text = (f"N={len(ray_i_lens['id'].unique())}" + "\n" +
                f"Slope={model.params['days']:.3f} mm/day")
        ax.text(0.95, 0.05, text, transform=ax.transAxes,
                ha="right", fontsize=6)
        ax.grid(alpha=0.3)
        ax.set_title(f"Ray {ray_idx}", fontsize=8)

    labels.append("Linear\nregression")
    handles.append(plt.Line2D([0], [0], color='black', linestyle='--'))
    fig.supxlabel("Days since first image")
    fig.supylabel("Ray length (mm)")
    fig.legend(handles=handles, labels=labels,
               title="Fish ID", fontsize=6, ncol=1, loc='outside right')

    plt.savefig("ray_len_over_time.pdf", bbox_inches='tight')
    plt.show()

    reg_slopes = pd.concat(reg_slopes)
    print(reg_slopes)

    fig, ax = plt.subplots()
    ax.scatter(reg_slopes["ray"], reg_slopes["slope"])
    ax.errorbar(reg_slopes["ray"], reg_slopes["slope"],
                yerr=[reg_slopes["slope"] - reg_slopes["ci_lower"],
                      reg_slopes["ci_upper"] - reg_slopes["slope"]],
                fmt='o', capsize=3)
    ax.set_title("Fin ray growth rates\nmixed effects model with 95% CI")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Ray index (from anterior to posterior)")
    ax.set_ylabel("Growth rate (mm/day)")
    plt.savefig("ray_growth_rates.pdf", bbox_inches='tight')
    plt.show()


def compress_pngs(dir_name):
    """Compress all PNG files in the specified directory using optipng."""
    png_files = [f for f in os.listdir(dir_name) if f.endswith('.png')]
    src_dir = dir_name
    dst_dir = dir_name + "_compressed"
    os.makedirs(dst_dir, exist_ok=True)

    sf = 0.1  # scale factor

    for fname in os.listdir(src_dir):
        if fname.lower().endswith(".png"):
            src_path = os.path.join(src_dir, fname)
            jpg_fname = os.path.splitext(fname)[0] + ".jpg"
            dst_path = os.path.join(dst_dir, jpg_fname)
            img = Image.open(src_path)
            w, h = img.size
            print(f"Resizing {fname} from {w}x{h} to "
                  + f"{int(w*sf)}x{int(h*sf)}")
            img = img.resize((int(w * sf), int(h * sf)), Image.LANCZOS)
            img = img.convert("RGB")  # JPEG does not support transparency
            img.save(dst_path, "JPEG", optimize=True, quality=95)


def print_help():
    """Print help message and exit."""
    help_text = """
    Usage: python img_handler.py [options]

    Options:
        -h              Show this help message and exit
        -d [directory]  Specify working directory
        -f [function]   Specify function to run (0-7)
                        0 ...load images from directory and count taxa
                        1 ...run landmarking GUI on images in directory
                        2 ...convert landmarks CSV to VTK files
                        3 ...visualise landmark VTK files
                        4 ...visualise landmarks layered plot
                        5 ...plot anterior-posterior length over time
                        6 ...plot fin ray length over time
                        7 ...compress PNG images in directory to small JPG
    """
    print(help_text)
    sys.exit()


if __name__ == "__main__":

    args = sys.argv[1:]
    if "-h" in args:
        print_help()

    if "-d" in args:
        WD = args[args.index("-d") + 1].rstrip("/")
    if "-f" in args:
        FUNC = int(args[args.index("-f") + 1])
        if FUNC == 0:
            img_paths = load_imgs_from_directory(WD)
            count_taxa(img_paths)
        elif FUNC == 1:
            lm = Landmarker(WD)
            lm.run(start_idx=START_FROM)
        elif FUNC == 2:
            landmarks_to_vtk(WD + "/lmks.csv",
                             wallnodes=False, landmarks=True)
        elif FUNC == 3:
            visualise_landmark_vtks(WD)
        elif FUNC == 4:
            visualise_landmarks_layered(WD)
        elif FUNC == 5:
            ap_len_over_time(WD + "/lmks.csv")
        elif FUNC == 6:
            ray_len_over_time(WD + "/lmks.csv")
        elif FUNC == 7:
            compress_pngs(WD)
        else:
            print_help()
    else:
        print_help()

        # dir_path = "wild_data"

        # img_paths = load_imgs_from_directory(dir_path)
        # count_taxa(img_paths)

        # lm = Landmarker("adult_benthic_all_images")
        # lm.run(start_idx=START_FROM)

        # landmarks_to_vtk("lmk_DA-1-10_12-09-25/landmarks.csv",
        #                  wallnodes=False, landmarks=True)
        # visualise_landmark_vtks("lmk_DA-1-10_12-09-25")

        # landmarks_to_vtk("lmks_all/lmks.csv")
        # visualise_landmark_vtks("lmks_all")
        # visualise_landmarks_layered("lmks_all")
        # compress_pngs("adult_benthic_all_images")
    # ap_len_over_time("lmks_all/lmks.csv")
