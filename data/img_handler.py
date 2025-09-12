"""For handling and retrieving data on real fin images."""

import os
import pandas as pd
import cv2


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
                if "landmarked" in file:
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
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            self.current_types.append(self.current_type)
            self.redraw_image()
            # Switch to edge after two scalebar points
            if len(self.current_types) >= 2 and self.current_types[-2:] == ["s", "s"]:
                self.current_type = "e"
                print("Automatically switched to edge landmark (green)")
            # Switch to fin after two edge points
            elif len(self.current_types) >= 4 and self.current_types[-2:] == ["e", "e"]:
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

    def run(self):
        """Run the landmarking process."""
        max_width, max_height = 2000, 1500
        for img_path in self.imgs["path"]:
            self.current_img_path = img_path
            img_blank = cv2.imread(img_path)
            img = cv2.imread(img_path)
            out_img_path = os.path.splitext(img_path)[0] + "_landmarked.png"
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
            print("Press 'f' for fin landmark (red), 'e' for edge landmark (green).")
            while True:
                cv2.imshow(img_path, self.current_img)
                key = cv2.waitKey(20)
                if key == ord('n'):  # Next image
                    self.landmarks[img_path] = list(
                        zip(self.current_points, self.current_types))
                    cv2.imwrite(out_img_path, self.current_img)
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
                    self.save_landmarks()
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


def landmarks_to_vtk(path):
    """Convert landmark points to VTK format for 3D visualization."""

    lmks = pd.read_csv(path)

    print(lmks)
    for fish in lmks["id"].unique():
        f_lmks = lmks[lmks["id"] == fish]
        for i, date in enumerate(f_lmks["date"].unique()):

            f_lmks_i = f_lmks[f_lmks["date"] == date]
            vtk_path = os.path.join(
                os.path.dirname(path), f"lmk_{fish}_{date}_{i}_landmarks.vtk")

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

            # Convert from pixels to mm (assuming 1mm scale bar)
            f_lmks_i = f_lmks_i.copy()  # Make an explicit copy
            f_lmks_i.loc[:, "x_mm"] = round(f_lmks_i["x"] / sb_len, 3)
            f_lmks_i.loc[:, "y_mm"] = round(f_lmks_i["y"] / sb_len, 3)

            # remove scale bar points
            f_lmks_i = f_lmks_i[f_lmks_i["type"] != "s"]

            with open(vtk_path, 'w', encoding='utf-8') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write(f"{fish} landmarks\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")
                f.write(f"POINTS {len(f_lmks_i)} float\n")
                for _, row in f_lmks_i.iterrows():
                    f.write(f"{row['x_mm']} {row['y_mm']} 0.0\n")
                f.write(f"POLYGONS 1 {len(f_lmks_i) + 1}\n")
                f.write(f"{len(f_lmks_i)} ")
                f.write(" ".join(str(j) for j in range(len(f_lmks_i))))
            print(f"VTK landmarks saved to {vtk_path}")


if __name__ == "__main__":
    # dir_path = "wild_data"

    # img_paths = load_imgs_from_directory(dir_path)
    # count_taxa(img_paths)

    # lm = Landmarker("adult_benthic_all_images")
    # lm.run()

    landmarks_to_vtk("lmk_DA-1-10_12-09-25/landmarks.csv")
