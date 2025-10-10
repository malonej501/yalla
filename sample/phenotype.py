import os
import sys
import math
import numpy as np
import pandas as pd
from vedo import Plotter, Points, Text2D, Axes, show, load, settings
from vedo import RendererFrame, Latex
from sklearn.cluster import DBSCAN
import alphashape
import shapely
from pyvirtualdisplay import Display
import h5py
import matplotlib
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from scipy.spatial import Delaunay, ConvexHull
from PIL import Image

matplotlib.use("pgf")
plt.style.use("../misc/stylesheet.mplstyle")

FUNC = 0
WD = "../run/saves/"


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
            f"../run/{self.run_id}/out_{self.wid}_{self.step}" +
            f"_{self.frame}.png")
        p.close()
        display.stop()


class Realfin():
    """Data from a real fin segmented to a .h5 file."""

    def __init__(self, path):
        self.path = path  # file name of the .h5 file
        self.id = os.path.basename(path).split(".")[0]
        with h5py.File(path, "r") as f:
            self.arr = np.squeeze(np.array(f["exported_data"]))
            self.total_area = self.arr.shape[0] * self.arr.shape[1]
        self.arr_lab = label(self.arr, background=2)  # label connected regions
        self.regions = regionprops(self.arr_lab)
        self.regions_sig = [  # remove very large and small regions
            r for r in self.regions if ((r.area < 0.5 * self.total_area) &
                                        (r.area > 0.001 * self.total_area))]

    def phenotype(self):
        """Returns the phenotype of the real fin."""

        if len(self.regions_sig) == 0:
            print(f"No significant regions found in {self.id}.")
            return {
                "id": self.id,
                "n_regions": 0,
                "mean_area": np.nan,
                "std_area": np.nan,
                "mean_roundness": np.nan,
                "std_roundness": np.nan
            }

        stats_full = []
        for r in self.regions_sig:  # only do stats for significant regions
            stats_full.append({
                "label": r.label,
                "n_regions": len(self.regions_sig),
                "area": r.area,
                "roundness": (4 * math.pi * r.area) / (r.perimeter**2),
            })

        stats_full = pd.DataFrame(stats_full)
        print(stats_full)
        mesh_stats, _, _ = self.mesh()
        return {
            "id": self.id,
            "n_regions": stats_full["n_regions"].iloc[0],
            "mean_area": stats_full["area"].mean(),
            "std_area": stats_full["area"].std(),
            "mean_roundness": stats_full["roundness"].mean(),
            "std_roundness": stats_full["roundness"].std(),
            "mesh_area": mesh_stats["mesh_area"],
            "av_dist": mesh_stats["av_dist"]
        }

    def plot_regions(self, display=False, export=False, ax=None):
        """Plots the regions marked as spots in the fin. Returns axis."""
        image_label_overlay = label2rgb(self.arr_lab,
                                        bg_color=(1, 1, 1))
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_label_overlay)

        for r in self.regions_sig:
            y, x = r.centroid
            ax.text(x, y - (self.arr.shape[1]*0.08), str(r.label),
                    fontsize=10, ha='center', va='center')

        ax.set_title(f"{self.id}")
        if export:
            plt.savefig(f"../data/{self.id}_regions.pdf", dpi=600)
        if display:
            plt.show()
        if fig is not None:
            plt.close(fig)
        return ax

    def mesh(self, display=False, ax=None):
        """Returns a mesh from the spot pattern centroids. If there are
        4 or more significant regions, a Delaunay triangulation is returned,
        3 returns convex hull, 2 a line and 1 a single point."""
        mesh = None
        area = 0
        av_dist = 0  # average distance between centroids
        centroids = np.array([r.centroid for r in self.regions_sig])

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.arr, cmap="gray")
        # ax.text(1, 1, f"N={len(centroids)}", ha='center', va='center')
        ax.set_title(self.id)

        if len(centroids) == 0:
            print(f"0 significant regions in {self.id}. Not "
                  + "enough to construct mesh.")
        if len(centroids) == 1:
            mesh = np.array([centroids[0, 1], centroids[0, 0]])
            ax.plot(centroids[0, 1], centroids[0, 0], "ro")
        if len(centroids) == 2:
            mesh = np.array([[centroids[0, 1], centroids[0, 0]],
                             [centroids[1, 1], centroids[1, 0]]])
            av_dist = np.linalg.norm(centroids[0, :2] - centroids[1, :2])
            ax.plot(centroids[:, 1], centroids[:, 0], color="red")
        if len(centroids) == 3:
            mesh = ConvexHull(centroids)
            area = mesh.volume
            for simplex in mesh.simplices:
                av_dist += np.linalg.norm(
                    centroids[simplex[0], :2] - centroids[simplex[1], :2])
                ax.plot(centroids[simplex, 1],
                        centroids[simplex, 0], color="red")
            av_dist /= len(mesh.simplices)  # calculate average distance
        if len(centroids) >= 4:
            mesh = Delaunay(centroids[:, :2])  # triangulate the points
            for simplex in mesh.simplices:
                av_dist += np.linalg.norm(
                    centroids[simplex[0], :2] - centroids[simplex[1], :2])
                pts = centroids[simplex, :2]
                # Shoelace formula for triangle area
                a = 0.5 * abs(
                    pts[0, 0]*(pts[1, 1]-pts[2, 1]) +
                    pts[1, 0]*(pts[2, 1]-pts[0, 1]) +
                    pts[2, 0]*(pts[0, 1]-pts[1, 1])
                )
            av_dist /= len(mesh.simplices)  # calculate average distance
            area += a
            ax.triplot(centroids[:, 1], centroids[:, 0],
                       mesh.simplices, color="red")
        ax.text(0.05, 0.05, f"N={len(centroids)}\narea={area:.2f}" +
                f"\nav_dist={av_dist:.2f}",
                transform=ax.transAxes, ha='left', va='bottom')

        if display:
            plt.show()
        if not display and fig is not None:
            plt.close(fig)  # ensure no figures left open

        stats = {
            "id": self.id,
            "n_regions": len(self.regions_sig),
            "mesh_area": area,
            "av_dist": av_dist
        }

        return stats, mesh, ax


def analyse_realfins(fin_dir="../data"):
    """Phenotype real fins in data directory."""

    bins = 10

    stats = []
    for file in os.listdir(fin_dir):
        if file.endswith(".h5"):
            fin = Realfin(path=os.path.join(fin_dir, file))
            stats.append(fin.phenotype())
            fin.plot_regions()

    stats = pd.DataFrame(stats)
    print(stats)

    plot_vars = ["n_regions", "mean_area",
                 "mean_roundness", "mesh_area", "av_dist"]
    plot_titles = [
        "No. Spots", "Mean Area", "Mean Roundness",
        "Mesh Area", "Average Centroid Separation Distance"
    ]
    fig, axs = plt.subplots(3, 2, figsize=(
        6, 6), layout="constrained", sharey=True)
    axs = axs.flat
    for i, plot_var in enumerate(plot_vars):
        axs[i].hist(stats[plot_var], bins)
        axs[i].set_xlabel(plot_titles[i])
        axs[i].grid(alpha=0.3)
    axs[-1].axis("off")  # hide the last empty subplot
    fig.supylabel("Frequency")
    fig.suptitle(fr"Real Fin Phenotype Statistics $N={len(stats)}$")

    plt.show()


def plot_segmented_fins(fin_dir="../data"):
    """Plots the segmented fins in the data directory."""
    for file in os.listdir(fin_dir):
        if file.endswith(".h5"):
            fin = Realfin(path=os.path.join(fin_dir, file))
            fin.plot_regions(display=False, export=True)


def compare_segmented_real(fin_dir="../data/data_23-06-25", export=False,
                           mode=0, nplot=0):
    """Plot segmented and real fins side by side.
    Args:
        fin_dir     directory containing the .h5 files of segmented fins.
        export      if True, saves the figure as a PDF.
        mode        0 segmented vs real
                    1 segmented + mesh vs real
        nplot       number of files to plot, if 0 all files are plotted.
    """
    hfiles = [f for f in os.listdir(fin_dir) if f.endswith(".h5")]
    pngs = [f.replace("_Simple Segmentation", "").replace(".h5", ".png")
            for f in hfiles]
    n = len(hfiles)
    assert len(pngs) == n, "Mismatch between .h5 and .png files."
    if n == 0:
        print("No .h5 files found.")
        return
    ncol = 1
    nrow = math.ceil(n / ncol) if nplot == 0 else math.ceil(nplot / ncol)
    _, axs = plt.subplots(nrow, ncol * 2, figsize=(
        1.5*ncol, 0.5 * nrow), layout="constrained")

    if n == 1:
        axs = [axs]  # Ensure axs is always iterable as a list of pairs
    axs = axs.reshape(-1, ncol * 2)

    for i, file in enumerate(hfiles):
        png = Image.open(os.path.join(fin_dir, pngs[i]))
        seg = Realfin(path=os.path.join(fin_dir, file))
        row = i // ncol
        col = (i % ncol) * 2
        axs[row][col].imshow(png)
        axs[row][col].axis("off")
        if mode == 0:
            seg.plot_regions(display=False, export=False, ax=axs[row][col + 1])
        elif mode == 1:
            seg.mesh(display=False, ax=axs[row][col + 1])
        axs[row][col + 1].set_title("")
        axs[row][col + 1].set_xticks([])
        axs[row][col + 1].set_yticks([])
        if nplot > 0 and i + 1 >= nplot:  # stop if reached nplot
            break
    if export:
        plt.savefig(f"real_seg_comp_{n}_{mode}.pdf", dpi=300)

    plt.show()


def tissue_properties(run_id):
    """Return properties of the tissue over the entire timecourse, not 
    deducible individual vtk files e.g. maximum no. cell types"""
    cell_types = []
    vtks = load(f"../run/{run_id}/out_0_0_*.vtk")
    for vtk in vtks:
        cell_types_i = np.unique(vtk.pointdata["cell_type"])
        if len(cell_types_i) > len(cell_types):
            cell_types = cell_types_i  # get the maximum no. cell types

    return cell_types


def plot_sim_tseries_vedo(run_id, n_frames, axes=False):
    """Plot a course time series of simulation frames."""
    settings.immediate_rendering = False
    cell_types = tissue_properties(run_id)
    frames = np.linspace(0, 100, n_frames, dtype=int)
    print(frames)
    p = Plotter(N=n_frames, size=(900, 600), bg="white")

    for i, fr in enumerate(frames):
        frame = Frame(run_id=run_id, wid=0, step=0, frame=fr)
        points = Points(frame.mesh).point_size(3)
        points.cmap("viridis", "cell_type", vmin=cell_types.min(),
                    vmax=cell_types.max())
        text = Text2D(txt=f"t={fr*10}", pos="top-middle")
        # text = Latex(formula=fr"t={fr*10}", pos=(0, 2, 0), s=2)
        p.at(i).add(points, text).add_renderer_frame()
        if axes:
            axes = Axes(xtitle="x", ytitle="y", text_scale=2,
                        xrange=(-3.75, 3.75), yrange=(-1.25, 1.25))
            p.at(i).add(axes)

    p.show()
    p.screenshot(f"{os.path.basename(run_id)}_tseries.png")
    p.interactive().close()


def plot_sim_tseries_mtpl(run_id, n_frames, nrow=2):
    """Plot a course time series of simulation frames using matplotlib."""
    cell_types = tissue_properties(run_id)
    frames = np.linspace(0, 100, n_frames, dtype=int)
    # frames = np.linspace(10, 70, n_frames, dtype=int)  # for wall-penetrating
    print(frames)
    ctypes = {1: "Spot-migratory", 2: "Non-spot", 3: "Spot-static"}

    if nrow == 1:
        fig, axs = plt.subplots(1, 6, figsize=(10, 1.5),
                                layout="constrained", sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(2, 3, figsize=(6, 3),
                                layout="constrained", sharex=True, sharey=True)
    axs = axs.flatten()

    for i, fr in enumerate(frames):
        frame = Frame(run_id=run_id, wid=0, step=0, frame=fr)
        fr_dat = {"x": frame.mesh.vertices[:, 0],
                  "y": frame.mesh.vertices[:, 1],
                  "type": frame.mesh.pointdata["cell_type"]}
        fr_dat = pd.DataFrame(fr_dat)
        for ctype in fr_dat["type"].unique():
            mask = fr_dat["type"] == ctype
            axs[i].scatter(fr_dat["x"][mask], fr_dat["y"][mask], s=1,
                           label=ctypes[ctype], c=plt.cm.viridis(
                               (ctype - cell_types.min()) /
                               (cell_types.max() - cell_types.min())),
                           rasterized=True)  # rasterize for large data
        axs[i].set_aspect("equal")
        axs[i].set_title(fr"$t={fr*10}$")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].axis("off")

    # fig.colorbar(sc, ax=axs, label="Cell Type", orientation="vertical")
    leg = fig.legend(*axs[3].get_legend_handles_labels(),
                     loc="outside lower center",
                     title="Cell Type", ncol=3)
    for handle in leg.legend_handles:
        handle.set_sizes([30])
        handle.set_edgecolor("black")
        handle.set_linewidth(0.5)
    # print(axs[4].get_legend_handles_labels())

    plt.savefig(f"{os.path.basename(run_id)}_tseries_mtpl.pdf")
    plt.show()


def print_help():
    help_text = """
    Usage: python phenotype.py [options]
    
    Options:
        -h              Show this help message and exit.
        -f [function]  Specify the function to run. Options include:
                       0 ...analyse_realfins
                       1 ...plot_sim_tseries_vedo
                       2 ...plot_sim_tseries_mtpl
                       3 ...plot_segmented_fins
                       4 ...compare_segmented_real
        -d [directory] Specify the directory contining data to plot.

    Description:
        This script provides functionalities to analyze and visualize 
        phenotypes of tissue simulations and real fin data. It includes 
        classes for handling simulation frames and real fin data, as well 
        as functions for plotting time series and comparing segmented 
        and real fins.
    """
    print(help_text)


if __name__ == "__main__":
    # collect bash arguments
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    if "-d" in args:
        WD = args[args.index("-d") + 1].rstrip("/")
    if "-f" in args:
        FUNC = int(args[args.index("-f") + 1])
        if FUNC == 0:
            analyse_realfins()
        elif FUNC == 1:
            plot_sim_tseries_vedo(WD, 6)
        elif FUNC == 2:
            plot_sim_tseries_mtpl(WD, 6)
        elif FUNC == 3:
            plot_segmented_fins()
        elif FUNC == 4:
            compare_segmented_real()
    # plot_segmented_fins()
    # compare_segmented_real(
    #     fin_dir="../data/data_23-06-25", export=True, mode=1, nplot=2)
    # Realfin(path="../data/1_13-10-22_Simple Segmentation.h5").mesh(True)
    # Realfin(path="../data/Taeniolethrinops_laticeps_'Tsano_Rock'_Simple Segmentation.h5").mesh(True)
    # Realfin(path="../data/Mchenga_cyclicos_'Msuli'_Simple Segmentation.h5").mesh(True)
    # print(stats)
    # plt.imshow(fin.arr, cmap="gray")
    # plt.show()
