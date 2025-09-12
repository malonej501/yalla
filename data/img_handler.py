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
    """Class for manually landmarking images using OpenCV.
    Start from the anterior edge of the fin and proceed clockwise.
    """

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.imgs = pd.DataFrame({"path": load_imgs_from_directory(dir_path)})
        self.imgs["id"] = self.imgs["path"].apply(
            lambda x: os.path.basename(x).split("_")[0])
        self.imgs["date"] = self.imgs["path"].apply(  # DD-MM
            lambda x: x.split("_")[-2])
        self.imgs["idx"] = self.imgs["path"].apply(
            lambda x: int(x.split("_")[-1].split(".")[0]))
        self.imgs = self.imgs.sort_values(
            by=["id", "idx"]).reset_index(drop=True)
        self.landmarks = {}  # {img_path: [(x1, y1), (x2, y2), ...]}
        self.current_points = []
        self.current_img = None
        self.current_img_path = None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events to record landmark points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            # img_copy = self.current_img.copy()
            for i, pt in enumerate(self.current_points):
                cv2.circle(self.current_img, pt, 5, (0, 0, 255), -1)
                cv2.putText(self.current_img, str(i+1), (pt[0]+8, pt[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(self.current_img_path, self.current_img)

    def run(self):
        """Main loop to display images and collect landmarks."""
        max_width, max_height = 2000, 1500
        for img_path in self.imgs["path"]:
            self.current_img_path = img_path
            img_blank = cv2.imread(img_path)  # blank image for undo
            img = cv2.imread(img_path)
            out_img_path = os.path.splitext(img_path)[0] + "_landmarked.png"
            self.current_points = []
            # Resize image if larger than max dimensions
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
            while True:
                cv2.imshow(img_path, self.current_img)
                key = cv2.waitKey(20)
                if key == ord('n'):  # Next image
                    self.landmarks[img_path] = self.current_points.copy()
                    cv2.imwrite(out_img_path, self.current_img)
                    break
                elif key == ord('r'):  # Remove last point
                    print(self.current_points)
                    self.current_img = img_blank.copy()
                    if self.current_points:
                        self.current_points.pop()
                    for i, pt in enumerate(self.current_points):
                        cv2.circle(self.current_img, pt, 5, (0, 0, 255), -1)
                        cv2.putText(self.current_img, str(i+1), (pt[0]+8, pt[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(img_path, self.current_img)
                elif key == ord('q'):  # Quit
                    self.landmarks[img_path] = self.current_points.copy()
                    cv2.imwrite(out_img_path, self.current_img)
                    cv2.destroyAllWindows()
                    self.save_landmarks()
                    return
            cv2.destroyAllWindows()
        self.save_landmarks()

    def save_landmarks(self, out_path="landmarks.csv"):
        """Save the collected landmarks to a CSV file."""
        import pandas as pd
        rows = []
        for img, pts in self.landmarks.items():
            for i, (x, y) in enumerate(pts):
                rows.append({"image": os.path.basename(img),
                            "landmark": i+1, "x": x, "y": y})
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"Landmarks saved to {out_path}")


if __name__ == "__main__":
    # dir_path = "wild_data"
    D = "adult_benthic_all_images"

    # img_paths = load_imgs_from_directory(dir_path)
    # count_taxa(img_paths)
    lm = Landmarker(D)
    lm.run()
