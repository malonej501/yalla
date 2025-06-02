import sys
import os
import math
import numpy as np
import pandas as pd
from vedo import Plotter, Points, show, addons, Text2D, Mesh, Line, Video
from vedo import load, image, pyplot
from sklearn.cluster import DBSCAN
import alphashape
import shapely
from pyvirtualdisplay import Display


class Frame():
    """Data from a single frame of a tissue simulation run."""

    def __init__(self, run_id, wid, step, frame):
        self.run_id = run_id  # run identifier
        self.wid = wid  # walk identifier
        self.step = step  # step number
        self.frame = frame  # frame number
        self.mesh = load(f"../run/{run_id}/out_{wid}_{step}_{frame}.vtk")
        print(f"../run/{run_id}/out_{wid}_{step}_{frame}.vtk")

    def phenotype(self, eps=0.05):
        """Returns the phenotype of the frame.
        args:
            eps: float, maximum distance between two samples for one to be
            considered as in the neighborhood of the other
        """
        stats = {
            "n_clusters": 0,
            "n_noise_pts": 0,
            "n_polygons": 0,
            "mean_area": np.nan,
            "std_area": np.nan,
            "mean_roundness": np.nan,
            "std_roundness": np.nan
        }
        # return positions of spot cells - both mobile and static
        x_spots = self.mesh.vertices[
            np.isin(self.mesh.pointdata["cell_type"], [1, 3])
        ]
        if x_spots.size == 0:  # skip if there no spot cells present in vtk
            return stats
        # eps - maximum distance between two samples for one to be considered
        # as in the neighborhood of the other - set to equilibrium distance
        # between spot cells - check force potential
        db = DBSCAN(eps=eps, min_samples=1).fit(x_spots)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        unique_labels = set(labels)
        stats["n_clusters"] = len(unique_labels) - (1 if -1 in labels else 0)
        # noisy points are given the label -1
        stats["n_noise_pts"] = list(labels).count(-1)
        # https://scikit-learn.org/1.5/modules/clustering.html#s
        # ilhouette-coefficient
        # s_coeff = metrics.silhouette_score(spot_cells, labels) # a higher
        # Silhouette Coefficient score relates to a model with better defined
        # clusters

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        areas = []
        # roundness is only computed for polygons - not lines, points etc.
        roundnesses = []
        n_poly = 0
        # return the coordinates of the cells in each cluster
        for l in unique_labels:
            class_member_mask = labels == l
            xy = x_spots[class_member_mask & core_samples_mask]
            xy = np.delete(xy, 2, axis=1)  # remove z dimension
            alpha_shape = alphashape.alphashape(xy, alpha=15)

            # calculate area for all types of geometries, not just polygons
            areas.append(alpha_shape.area)
            # or alpha_shape.geom_type == "MultiPolygon":
            if alpha_shape.geom_type == "Polygon":
                n_poly += 1  # count number of polygons
                perimeter = shapely.length(alpha_shape)
                # 1 for perfect circle, 0 for non-circular
                roundnesses.append(
                    (4 * math.pi * alpha_shape.area) / (perimeter**2))
            if alpha_shape.geom_type == "LineString":
                roundnesses.append(0)  # if line roundness is 0
            if alpha_shape.geom_type == "Point":
                roundnesses.append(1)  # if point roundness is 1
        stats["n_polygons"] = n_poly
        stats["mean_area"] = np.mean(areas)
        stats["std_area"] = np.std(areas)
        stats["mean_roundness"] = np.mean(roundnesses)
        stats["std_roundness"] = np.std(roundnesses)

        return stats

    def render(self, zoom=0.6, pt__size=12, c_prop="cell_type"):
        """Generates a .png of the frame."""
        # virtual display for offscreen rendering
        display = Display(visible=0, size=(1366, 768))
        display.start()
        cmap = "viridis"
        points = Points(self.mesh).point_size(
            pt__size * zoom).cmap(cmap, c_prop)
        p = show(points, interactive=False)
        p.screenshot(
            f"../run/{self.run_id}/out_{self.wid}_{self.step}_{self.frame}.png")
        p.close()
        display.stop()
