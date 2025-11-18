# For analysing the output of ilastic cell segementations
import os
import sys
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
from pyparsing import cached_property
import cv2
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import networkx as nx

FUNC = 0  # function to run
PROB_THRESH = 0.5  # threshold for probability maps
REG_AREA_THRESH = 5  # minimum area (in pixels) for segmented regions
# set to None to disable area filtering
K = 3  # default number of neighbors for k-NN graph
PT_SIZE = 10  # point size for plotting centroids
# WD = "adult_benthic_all_images"
# WD = "adult_benthic_training_irid"
WD = "deep_adults_sorted"
LMK_PTH = "deep_adults_sorted_BT_16-11-25.csv"
FISH_ID = 6  # ID of fish to analyse 0-9
STAGE = 15  # 0-21
R_RAD = 0.085  # radius for neighbour counts within radius (mm)
A_RAD = 0.5  # inner radius for annulus neighbour counts (mm)
A_DELTA = 0.025  # width of annulus for annulus neighbour counts (mm)

# matplotlib.use("pgf")
# plt.style.use("../misc/stylesheet.mplstyle")


class Cell_seg:  # short for segmentation
    """Class for handling multiple fin segmentations in a directory."""

    def __init__(self, wd):
        self.wd = wd
        self.hfiles = [f for f in os.listdir(wd) if f.endswith(".h5")]
        self.img_files = [f.replace("_Probabilities.h5", ".tiff")
                          for f in self.hfiles]
        self.metadata = self.get_seg_data()
        self.lmks = pd.read_csv(LMK_PTH)
        self.assign_scale_bars()
        self.fins = [
            Fin(row["file"], row["img"], wd,
                sb_len=row["sb_len_px"])
            for _, row in self.metadata.iterrows()
        ]

    def get_seg_data(self):
        """Extract metadata from filenames and return as DataFrame."""
        df = []
        for hfile in self.hfiles:
            img_file = hfile.replace("_Probabilities.h5", ".tiff")
            fish_id = hfile.split("_")[0]
            date = hfile.split("_")[1]
            # stage = int(hfile.split("_")[2])
            type = "probs" if "Probab" in hfile else "simple"
            df.append({"file": hfile, "img": img_file, "id": fish_id,
                       "date": date,  # "stage": stage,
                       "type": type})

        df = pd.DataFrame(df)
        df["date"] = pd.to_datetime(df["date"])  # , format="%d-%m")
        df["date"] = df["date"].apply(lambda x: x.replace(year=2022))
        df["stage"] = df["date"].rank(method="dense").astype(int)
        df["day"] = (df["date"] - df["date"].min()).dt.days
        df.sort_values(by=["id", "day"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def assign_scale_bars(self):
        """Assign scale bar lengths to each fin based on landmark data."""
        for fish_id, stage in self.metadata[
                ["id", "stage"]].itertuples(index=False):
            lmks_fish = self.lmks[(self.lmks["id"] == fish_id) &
                                  (self.lmks["idx"] == stage) &
                                  (self.lmks["type"] == "s")]
            if lmks_fish.empty:
                print(
                    f"Warning: No scale bar landmarks for {fish_id} stage {stage}")
                sb_len = np.nan
            else:
                p0 = lmks_fish.iloc[0][["x", "y"]].to_numpy(dtype=float)
                p1 = lmks_fish.iloc[1][["x", "y"]].to_numpy(dtype=float)
                sb_len = np.linalg.norm(p1 - p0)
            self.metadata.loc[
                (self.metadata["id"] == fish_id) &
                (self.metadata["stage"] == stage), "sb_len_px"] = sb_len


class Fin:
    """Class representing a single fin segmentation."""

    def __init__(self, hfile, img_file, wd, sb_len: float = None):
        self.hfile = hfile
        self.img_file = img_file
        self.img_pth = os.path.join(os.path.join(wd, ".."), img_file)
        self.wd = wd
        self.path = os.path.join(wd, hfile)
        self.fish_id = self.hfile.split("_")[0]
        self.date = self.hfile.split("_")[1]
        self.stage = self.hfile.split("_")[2]
        self.type = "probs" if "Probab" in self.hfile else "simple"
        self.sb_len = sb_len if not np.isnan(sb_len) else 1

    @cached_property  # to avoid loading all files at init
    def array(self) -> np.ndarray:
        """Load HDF5 dataset on first access and cache it."""
        with h5py.File(self.path, "r") as hf:
            data = np.squeeze(np.array(hf["exported_data"]))
        return np.asarray(data)

    @cached_property
    def arr_lab(self) -> np.ndarray:
        """Labelled array of segmented regions."""
        arr_bin = (self.array[:, :, 0] > PROB_THRESH).astype(int)
        arr_lab = label(arr_bin, background=1)
        return arr_lab

    @cached_property
    def regions(self) -> np.ndarray:
        """Get unique segmented regions in the array."""
        regions = regionprops(self.arr_lab)
        if REG_AREA_THRESH is not None:
            regions = [r for r in regions if r.area > REG_AREA_THRESH]
        return regions

    def centroids(self) -> np.ndarray:
        """Get centroids of segmented regions as (x, y) array."""
        cent = np.array([r.centroid for r in self.regions])
        cent /= self.sb_len  # scale centroids from px to mm
        cent[:, 0] *= -1  # invert y-axis
        return cent[:, ::-1]  # convert (row, col) to (x, y)

    def plot_probs(self):
        """Plot the raw probability map."""
        plt.imshow(self.array[:, :, 1], cmap="gray_r")
        plt.title(self.hfile)
        plt.colorbar(label="Probability")
        plt.show()

    def plot_seg(self):
        """Plot the segmented regions."""
        # original_img = plt.imread(os.path.join(self.wd, self.img_file))
        original_img = cv2.imread(self.img_pth)
        img_overlay = label2rgb(
            self.arr_lab, image=original_img, bg_label=0)

        # plt.imshow(original_img, alpha=0.5)
        plt.imshow(img_overlay)
        plt.title(f"Segmented Regions: {self.hfile}")
        plt.show()

    def delaunay(self):
        """Compute Delaunay triangulation of centroids."""
        cent = self.centroids()
        delaunay = DelaunayTri(cent, self.sb_len, self.hfile, self.img_pth)
        return delaunay

    def knn_graph(self, k=K):
        """Compute k-nearest neighbor graph of centroids."""
        cent = self.centroids()
        knn = KNNGraph(cent, self.sb_len, k, self.hfile, self.img_pth)
        return knn

    def counts_within_radius(self, r: float):
        """Count number of neighbors within radius r for each centroid."""
        pts = self.centroids()
        nbrs = NearestNeighbors(radius=r).fit(pts)
        _, indices = nbrs.radius_neighbors(pts, return_distance=True)
        counts = np.array([len(ind) - 1 for ind in indices])  # exclude self

        return counts

    def counts_within_annulus(self, r_in: float, delta: float):
        """Count no. neighbours within annulus with inner radius r_in
            and outer radius r_in + delta."""
        pts = self.centroids()
        nbrs_out = NearestNeighbors(radius=r_in+delta).fit(pts)
        _, i_out = nbrs_out.radius_neighbors(pts, return_distance=True)
        counts_out = np.array([len(i) - 1 for i in i_out])
        nbrs_in = NearestNeighbors(radius=r_in).fit(pts)
        _, i_in = nbrs_in.radius_neighbors(pts, return_distance=True)
        counts_in = np.array([len(i) - 1 for i in i_in])

        counts = counts_out - counts_in

        return counts

    def plot_nb_counts(self, fig=None, axs=None, vmin=None, vmax=None):
        """Visualise neighbour counts for real fin."""
        r_rad = R_RAD  # 0.085  # mm
        a_rad = A_RAD  # 0.5
        a_delta = A_DELTA  # 0.025

        r_cts = self.counts_within_radius(r_rad)
        a_cts = self.counts_within_annulus(a_rad, a_delta)
        x, y = self.centroids()[:, 0], self.centroids()[:, 1]

        titles = [rf"$N_\beta = s < {r_rad}$ (mm)",
                  rf"$N_\gamma = {a_rad} < s < {a_rad + a_delta}$ (mm)"]

        img = plt.imread(self.img_pth)
        extent = [0, img.shape[1]/self.sb_len, -img.shape[0]/self.sb_len, 0]

        if fig is None or axs is None:
            fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(8, 8),
                                    layout="constrained",
                                    sharex=True, sharey=True)
        scs, cbs = [], []
        for i, c_scale in enumerate([r_cts, a_cts]):
            ax = axs.flatten()[i]
            sc = ax.scatter(x, y, c=c_scale, cmap="viridis",
                            s=PT_SIZE, vmin=vmin, vmax=vmax)
            scs.append(sc)
            cb = fig.colorbar(sc, ax=ax, shrink=0.5, location="bottom")
            cbs.append(cb)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_title(self.hfile + "\n" + titles[i])

            ax.imshow(img, extent=extent)
        fig.supxlabel("AP (mm)")
        fig.supylabel("PD (mm)")

        # pos = (1.0, -2.5)
        pos = (x.max() * 0.15, y.min() * 0.85)

        r = Circle(pos, radius=r_rad, color="C1", alpha=0.5, ec=None)
        a = Wedge(pos, a_rad+a_delta, 0, 360,
                  width=a_delta, fc="C1", ec="None", alpha=0.5)
        axs[0].add_patch(r)
        axs[1].add_patch(a)

        if fig is None or axs is None:
            plt.show()
        else:
            return fig, axs, scs, cbs


class DelaunayTri():
    """Delaunay triangulation of points."""

    def __init__(self, points: np.ndarray, sb_len: float, hfile: str = None,
                 img_pth: str = None):
        self.points = points
        self.tri = Delaunay(points)
        self.hfile = hfile
        self.img_pth = img_pth
        self.sb_len = sb_len
        self.get_edge_distances()

    def get_edge_distances(self):
        """Compute distances between connected points without duplication"""
        edges = set()
        for tri in self.tri.simplices:
            i0, i1, i2 = tri
            edges.add((min(i0, i1), max(i0, i1)))
            edges.add((min(i1, i2), max(i1, i2)))
            edges.add((min(i2, i0), max(i2, i0)))

        # compute lengths
        distances = np.array(
            [np.linalg.norm(self.points[i] - self.points[j]
                            ) for i, j in edges], dtype=float)

        print(distances)
        print(distances.mean())
        plt.hist(distances, bins=200)
        plt.xlabel("Edge length (mm)")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        # plt.xlim(0, 1)
        plt.show()

    def plot(self):
        """Plot the Delaunay triangulation."""
        fig, ax = plt.subplots(layout="constrained")
        ax.triplot(self.points[:, 0], self.points[:, 1], self.tri.simplices,
                   lw=0.5)

        img = plt.imread(os.path.join(self.img_pth))
        extent = [0, img.shape[1]/self.sb_len, -img.shape[0]/self.sb_len, 0]
        ax.imshow(img, alpha=0.5, extent=extent)

        ax.set_xlabel("AP (mm)")
        ax.set_ylabel("PD (mm)")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        ax.set_title(f"Delaunay Triangulation\n{self.hfile}")
        plt.show()


class KNNGraph():
    """k-nearest neighbor graph of points."""

    def __init__(self, points: np.ndarray, sb_len: float, k: int, hfile: str = None,
                 img_pth: str = None):
        self.points = points
        self.sb_len = sb_len if not np.isnan(sb_len) else 1
        self.k = k
        self.edges, self.distances = self.compute_knn()
        self.unique_distances = [
            np.linalg.norm(self.points[i] - self.points[j])
            for i, j in self.edges]
        self.hfile = hfile
        self.img_pth = img_pth
        self.av_dist = np.mean(self.distances[:, 1:])  # exclude self-distance

    def compute_knn(self):
        """Compute k-nearest neighbor edges."""
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(self.points)
        distances, indices = nbrs.kneighbors(self.points, return_distance=True)
        # Build adjacency list
        edges = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:
                if i != j:  # skip self-loops
                    edges.append((i, j))

        print(distances)
        return edges, distances

    def compute_nx_graph(self):
        """Compute NetworkX graph from k-nearest neighbor edges."""
        G = nx.Graph()
        G.add_edges_from((edge[0], edge[1], {
                         'dist': self.unique_distances[i]}
        ) for i, edge in enumerate(self.edges))
        connected = list(nx.connected_components(G))
        return G, connected

    def plot(self):
        """Plot the k-nearest neighbor graph."""
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        G, connected = self.compute_nx_graph()
        connected.sort(key=len, reverse=True)  # sort components by size
        pos = {i: self.points[i] for i in range(len(self.points))}
        colors = plt.get_cmap('tab10', len(connected))

        av_dists = []
        for i, component in enumerate(connected):
            subgraph = G.subgraph(component)
            # compute the total distance of edges in subgraph / total no. edges
            av_dist = subgraph.size(weight="dist") / subgraph.number_of_edges()
            av_dists.append(av_dist)
            nx.draw_networkx_edges(
                subgraph, pos=pos, ax=ax, edge_color=colors(i), width=0.5,
                hide_ticks=False,
                label=f"{i}: n={
                    len(component)}, avg. dist={
                        av_dist:.2f} mm" if i < 10 else None)
        ax.set_aspect("equal")

        img = plt.imread(os.path.join(self.img_pth))
        extent = [0, img.shape[1]/self.sb_len, -img.shape[0]/self.sb_len, 0]
        ax.imshow(img, alpha=0.5, extent=extent)

        fig.legend(loc="outside right")
        ax.set_xlabel("AP (mm)")
        fig.supylabel("PD (mm)")
        ax.grid(alpha=0.3)
        ax.set_title(f"k-NN Graph (k={self.k})\n{self.hfile} ")

        plt.show()


class nb_counts_animation():
    """Animate neighbour counts over stages for a given fish ID."""

    def __init__(self, wd, fish_id: int):
        self.wd = wd
        self.fish_id = fish_id
        self.seg = Cell_seg(wd)
        self.fish_dat = self.seg.metadata[self.seg.metadata["id"]
                                          == f"DA-{1+fish_id}"]
        self.fig, self.ax = plt.subplots(ncols=1, nrows=2, figsize=(8, 8),
                                         layout="constrained",
                                         sharex=True, sharey=True)
        self.scs = []
        self.cbs = []

    def init(self):
        self.fig, self.ax, self.scs, self.cbs = self.seg.fins[0].plot_nb_counts(
            fig=self.fig, axs=self.ax)

    def update(self, frame):
        # stage = self.fish_dat.iloc[frame]["stage"]
        fin = self.seg.fins[frame]
        img = plt.imread(fin.img_pth)
        extent = [0, img.shape[1]/fin.sb_len, -img.shape[0]/fin.sb_len, 0]
        for ax in self.ax.flatten():
            ax.images[0].set_array(img)
            ax.images[0].set_extent(extent)

        r_cts = fin.counts_within_radius(R_RAD)
        a_cts = fin.counts_within_annulus(A_RAD, A_DELTA)
        print(max(r_cts), max(a_cts))

        self.scs[0].set_offsets(fin.centroids())
        self.scs[1].set_offsets(fin.centroids())
        self.scs[0].set_array(r_cts)
        self.scs[1].set_array(a_cts)
        self.scs[0].set_clim(0, max(r_cts))
        self.scs[1].set_clim(0, max(a_cts))
        self.cbs[0].update_normal(self.scs[0])
        self.cbs[1].update_normal(self.scs[1])

        self.ax[0].set_title(fin.hfile + "\n" +
                             rf"$N_\beta = s < {R_RAD}$ (mm)")
        self.ax[1].set_title(fin.hfile + "\n" +
                             rf"$N_\gamma = {A_RAD} < s < {A_RAD + A_DELTA}$ (mm)")

        # pass

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, frames=range(len(self.fish_dat)),
                            init_func=self.init, blit=False, interval=500, repeat=False)
        ani.save(f"nb_counts_fish{self.fish_id}.mp4", dpi=300)
        # plt.show()


def print_help():
    """Print help message and exit."""
    help_text = """
    Usage: python cell_seg.py [options]

    Options:
        -h              Show this help message and exit
        -d [directory]  Specify working directory
        -f [function]   Specify function to run
                        0 ... Plot k-NN graph for one fin
                        1 ... Plot Delaunay triangulation for one fin
                        2 ... Plot neighbour counts for one fin
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
            seg = Cell_seg(WD)
            idx = seg.metadata[seg.metadata["id"] ==
                               f"DA-{1+FISH_ID}-1{FISH_ID}"]
            idx = idx[idx["stage"] == STAGE]
            idx = idx.index[0]
            seg.fins[idx].knn_graph().plot()
        elif FUNC == 1:
            seg = Cell_seg(WD)
            idx = seg.metadata[seg.metadata["id"] ==
                               f"DA-{1+FISH_ID}-1{FISH_ID}"]
            idx = idx[idx["stage"] == STAGE]
            idx = idx.index[0]
            seg.fins[idx].delaunay().plot()
        elif FUNC == 2:
            seg = Cell_seg(WD)
            print(seg.metadata)
            idx = seg.metadata[seg.metadata["id"] ==
                               f"DA-{1+FISH_ID}"]
            idx = idx[idx["stage"] == STAGE]
            idx = idx.index[0]
            seg.fins[idx].plot_nb_counts()
        elif FUNC == 3:
            seg = Cell_seg(WD)
            seg.fins[35].plot_probs()
        elif FUNC == 4:
            ani = nb_counts_animation(WD, FISH_ID)
            ani.animate()
    else:
        print_help()
