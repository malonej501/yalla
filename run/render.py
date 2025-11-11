import sys
import os
import shapely
import math
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from vedo import Plotter, Points, show, addons, Text2D, Mesh, Line, Video
from vedo import load, image, pyplot, Axes, Glyph, Arrow, NormalLines
import pandas as pd
import numpy as np
import alphashape
from sklearn.cluster import DBSCAN
from pyvirtualdisplay import Display
from scipy.spatial.distance import pdist

# Default parameters

# Visualisation
EXPORT = False  # export the rendering to video file
C_PROP = "cell_type"  # cell property to colourise
FUNC = 0  # by default render movie
ZOOM = 0.6  # define the how far the camera is out
PT_SIZE = 12  # how large the cells are drawn
PA = 0.7  # point alpha
FA = 0.1  # fin alpha
RA = 0.1  # ray alpha
ANIMATE = 2  # 0 = False, 1 = Matplotlib, 2 = Vedo
SHOW_AX = True  # show axes
CELLS = True  # render cells if present
WALLS = True  # render walls if present
FIN = True  # render fin mesh if present
RAYS = True  # render ray mesh if present
FOLDER_PATH = "../run/saves/test"  # default output folder
VTKS = None  # list of vtk files
WALK_ID = 0  # defualt if only one tissue simulation
STEP = 0

# DBSCAN parameters
EPS = 0.05  # maximum distance between two samples for one to be considered
# as in the neighborhood of the other


def render_frame():
    """Render the final element of a list of vtks offscreen and export as
    .png"""

    # virtual display for offscreen rendering
    display = Display(visible=0, size=(1366, 768))
    display.start()
    vtk = VTKS[-1]  # select final frame
    cmap = "viridis"
    points = Points(vtk).point_size(PT_SIZE * ZOOM).cmap(cmap, C_PROP)
    p = show(points, interactive=False)
    p.screenshot(f"{FOLDER_PATH}/out_{WALK_ID}_{STEP}_{len(VTKS)-1}.png")
    p.close()
    display.stop()


def render_movie(walls=False, fin=False, cells=True, rays=False):
    """Renders movie of growing tissue with one cell property colourised
    e.g. cell_type, u, mech_str"""

    print(f"Rendering: {FOLDER_PATH}")

    video_length = 10  # in seconds
    cmap = "viridis"
    # load global properties for consistent gui throught timecourse
    cell_types, max_bounds = tissue_properties()

    # Create a plotter
    p = Plotter(interactive=False)
    p.zoom(ZOOM)
    axes = Axes(xtitle="x", ytitle="y", ztitle="z",
                xrange=(max_bounds[0], max_bounds[1]),
                yrange=(max_bounds[2], max_bounds[3]))
    p.add(axes)

    if EXPORT:
        v = Video(
            name=f"../run/saves/{FOLDER_PATH.rsplit('/', maxsplit=1)[-1]}" +
            f"_{C_PROP}.mp4",
            duration=video_length,
            backend="imageio")

    no_frames = min(len(VTKS), len(W_VTKS) if W_VTKS is not None else np.nan,
                    len(F_VTKS) if F_VTKS is not None else np.nan,
                    len(R_VTKS) if R_VTKS is not None else np.nan)

    frames = []
    # Loop through the VTK files and visualize them
    for i in range(no_frames):

        pts, wpts = Points([]), Points([])  # empty points object
        fmesh, rmesh = Mesh(), Mesh()  # empty mesh object
        if cells:
            pts = Points(VTKS[i]).alpha(PA).point_size(
                PT_SIZE * ZOOM)  # originally 10
        if walls:
            wpts = Points(W_VTKS[i]).point_size(PT_SIZE * ZOOM).color("black")
        wnrms = Glyph(wpts, Arrow().scale(0.5), "normals", c="blue")
        if fin:
            fmesh = Mesh(F_VTKS[i]).alpha(
                FA).linecolor("black").color("grey")  # .wireframe()
        if rays:
            rmesh = Mesh(R_VTKS[i]).alpha(
                RA).linecolor("black").color("grey")  # .wireframe()

            # lims = ((pts.bounds()[0],pts.bounds()[1]),
            # (pts.bounds()[2],pts.bounds()[3]))
        if C_PROP == "cell_type" and cells:  # ensure cmap for c_type is constant
            pts.cmap(cmap, C_PROP, vmin=min(cell_types),
                     vmax=max(cell_types))
        elif cells:
            pts.cmap(cmap, C_PROP)
        br = addons.ScalarBar(pts, title=C_PROP)

        info_str = f"i: {i} (day {i*2})"
        if cells:
            info_str = (
                f"i: {i} (day {i*2})\n"
                f"n: {len(pts.vertices)}\n" +
                "".join([
                    f"n_{cell_type}: {len(
                        pts.pointdata['cell_type'][
                            pts.pointdata['cell_type'] == cell_type])}\n"
                    for cell_type in cell_types
                ])
            )
        info = Text2D(txt=info_str, pos="bottom-left")

        pts.name = "cells"
        wpts.name = "wall_nodes"
        wnrms.name = "wall_normals"
        fmesh.name = "fin_mesh"
        rmesh.name = "ray_mesh"
        if cells:
            br.name = "bar"
        info.name = "info"
        frames.append((pts, wpts, wnrms, fmesh, rmesh, br, info))
        # Add the mesh to the plotter
        p.remove("cells").add(pts)
        p.remove("wall_nodes").add(wpts)
        p.remove("wall_normals").add(wnrms)
        p.remove("fin_mesh").add(fmesh)
        p.remove("ray_mesh").add(rmesh)
        p.remove("info").add(info)
        p.remove("bar").add(br)
        # if i == 0:  # only add bar once to avoid flickering
        #     p.add(br)
        p.show(zoom="tight")  # now means auto axs

        if EXPORT:
            v.add_frame()

    if EXPORT:
        v.close()

    def slider1(widget, _):
        val = widget.value  # get the slider current value
        pts, wpts, wnrms, fmesh, rmesh, br, info = frames[int(val)]

        p.remove("cells").add(pts)
        p.remove("wall_nodes").add(wpts)
        p.remove("wall_normals").add(wnrms)
        p.remove("fin_mesh").add(fmesh)
        p.remove("ray_mesh").add(rmesh)
        p.remove("bar").add(br)
        p.remove("info").add(info)

        p.render()

    def slider2(widget, _):
        val = int(widget.value)  # get slider value
        if not hasattr(slider2, "prev_val"):
            slider2.prev_val = val  # initialise prev_val
        if val == slider2.prev_val:  # only update if int val has changed
            return
        slider2.prev_val = val
        # get current frame
        pts, wpts, wnrms, fmesh, rmesh, br, info = frames[int(val)]
        c_prop_local = pts.pointdata.keys()[val]  # return new cmap from slider

        # change the cmap for and bar to the current frame
        pts.cmap(cmap, c_prop_local)
        br = addons.ScalarBar(pts, title=c_prop_local)
        br.name = "bar"
        p.remove("bar").add(br)
        p.render()

        # change the cmap for and add bar to all frames
        for k, (pts, wpts, wnrms, fmesh, rmesh, br, info) in enumerate(frames):
            pts = pts.cmap(cmap, c_prop_local)
            br = addons.ScalarBar(pts, title=c_prop_local)
            br.name = "bar"
            frames[k] = (pts, wpts, wnrms, fmesh, rmesh, br, info)

    p.add_slider(slider1, 0, len(frames)-1, pos="top-right", value=len(frames))
    if cells:
        p.add_slider(slider2, 0, len(pts.pointdata.keys())-1,
                     pos="top-left", value=pts.pointdata.keys().index(C_PROP))
    p.interactive()


def show_chem_grad():
    """Plot chemical amount along x axis of tissue for several timepoints"""
    pts = load_vtks(FOLDER_PATH)

    n_t = 10

    print(pts[0])
    print(pts[0].pointdata["cell_type"])
    print(pts[0].vertices)
    print(pts[0].pointdatafmesh["u"])
    x_pos = pts[0].vertices[:, 0]

    u = pts[0].pointdata["u"]
    v = pts[0].pointdata["v"]
    attributes = [u, v]

    _, axs = plt.subplots(n_t, 2, figsize=(10, 2*n_t))

    for i, row in enumerate(axs):
        t = int(i * len(pts)/n_t)
        u = pts[t].pointdata["u"]
        v = pts[t].pointdata["v"]
        attributes = [u, v]
        for j, ax in enumerate(row):
            ax.scatter(x_pos, attributes[j])
            ax.set_title(f"t = {t}")
            ax.set_xlabel("x_pos")
            if j == 0:
                ax.set_ylabel("u")
            if j == 1:
                ax.set_ylabel("v")

    plt.tight_layout()
    plt.show()


def tissue_stats():
    """Return basic tissue stats for each vtk file e.g. tissue xmin and xmax,
    no. each cell type etc."""

    stats = []
    for pt in VTKS:
        # print(pt.vertices[:10])
        xmax = max(pt.vertices[:, 0])
        ymax = max(pt.vertices[:, 1])
        xmin = min(pt.vertices[:, 0])
        ymin = min(pt.vertices[:, 1])

        n_a = len(pt.pointdata["cell_type"][pt.pointdata["cell_type"] == 1])
        n_b = len(pt.pointdata["cell_type"][pt.pointdata["cell_type"] == 2])

        stats.append({
            'xmax': xmax,
            'ymax': ymax,
            'xmin': xmin,
            'ymin': ymin,
            'n_A': n_a,
            'n_B': n_b,
        })

    stats_df = pd.DataFrame(stats)
    print(stats_df)


def pattern_stats(fin=False, rays=False):
    """Cluster spot cells and infer alpha shapes and shape stats"""

    cell_types, max_bounds = tissue_properties()

    stats = []
    frames = []
    print(f"Clustering and alpha shapes analysis: {FOLDER_PATH}")
    for i, mesh in enumerate(VTKS):
        sys.stdout.write(f"\rProcessing frame: {i}")
        sys.stdout.flush()
        mesh = VTKS[i]
        # return positions of spot cells
        cell_types = mesh.pointdata["cell_type"]
        mask = (cell_types == 1) | (cell_types == 3)
        x_spots = mesh.vertices[mask]
        if x_spots.size == 0:  # skip if there no spot cells present in vtk
            continue
        # eps - maximum distance between two samples for one to be considered
        # as in the neighborhood of the other - set to equilibrium distance
        # between spot cells - check force potential
        db = DBSCAN(eps=EPS, min_samples=1).fit(x_spots)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)  # noisy points are given the label -1
        # https://scikit-learn.org/1.5/modules/clustering.html#s
        # silhouette-coefficient
        # s_coeff = metrics.silhouette_score(spot_cells, labels) # a higher
        # Silhouette Coefficient score relates to a model with better defined
        # clusters

        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        # roundness is only computed for polygons - not lines, points etc.
        a_shapes, areas, roundnesses, max_diams, tags = [], [], [], [], []
        a_meshes, p_meshes = [], []  # alpha shapes & spot cell points
        n_poly = 0
        # return the coordinates of the cells in each cluster
        for l in unique_labels:
            class_member_mask = labels == l
            xy = x_spots[class_member_mask & core_samples_mask]
            p_mesh = Points(xy).point_size(PT_SIZE * ZOOM).color(l)
            p_mesh.name = "p_mesh"
            p_meshes.append(p_mesh)
            xy = np.delete(xy, 2, axis=1)  # remove z dimension
            alpha_shape = alphashape.alphashape(xy, alpha=15)

            # plot alpha shape with Polygon(list(alpha_shape.exterior.coords),
            # alpha=0.2)
            a_shapes.append((l, alpha_shape))
            # calculate area for all types of geometries, not just polygons
            areas.append(alpha_shape.area)
            # or alpha_shape.geom_type == "MultiPolygon":
            if alpha_shape.geom_type == "Polygon":
                n_poly += 1  # count number of polygons
                a_shape_np = np.array(alpha_shape.exterior.coords)
                z_col = np.zeros((a_shape_np.shape[0], 1))
                a_shape_np = np.hstack((a_shape_np, z_col))
                cells = np.array([range(len(a_shape_np))])

                a_mesh = Mesh([a_shape_np, cells])
                # only display label if polygon
                a_mesh_ids = a_mesh.labels(
                    content=np.array([l]), on="cells", yrot=180)
                a_mesh_ids.name = "tag"
                tags.append(a_mesh_ids)
                a_mesh.color(l).alpha(0.2)
                a_mesh.triangulate()  # ensure neat rendering of final shapes
                a_mesh.name = "a_mesh"
                a_meshes.append(a_mesh)

                perimeter = shapely.length(alpha_shape)
                # 1 for perfect circle, 0 for non-circular
                roundnesses.append(
                    (4 * math.pi * alpha_shape.area) / (perimeter**2))

                max_diam = pdist(alpha_shape.exterior.coords).max()
                max_diams.append(max_diam)

            if alpha_shape.geom_type == "LineString":
                a_meshes.append(None)
                tags.append(None)
                roundnesses.append(0)  # if line roundness is 0
                max_diams.append(np.nan)
            if alpha_shape.geom_type == "Point":
                a_meshes.append(None)
                tags.append(None)
                roundnesses.append(1)  # if point roundness is 1
                max_diams.append(np.nan)
        info = Text2D(
            txt=(f"i: {i}\n"
                 f"n_clusters: {len(unique_labels)}\n"
                 f"n_polygons: {n_poly}\n"
                 f"mean_area: {np.nanmean(areas):.4f}\n"
                 f"std_area: {np.nanstd(areas):.4f}\n"
                 f"mean_roundnesses: {np.nanmean(roundnesses):.4f}\n"
                 f"std_roundnesses: {np.nanstd(roundnesses):.4f}\n"),
            pos="bottom-left")
        info.name = "info"

        fmesh = Mesh()  # empty mesh object
        if fin:
            fmesh = Mesh(F_VTKS[i]).alpha(
                0.1).linecolor("black").color("grey")
            fmesh.name = "fin_mesh"
        if rays:
            rmesh = Mesh(R_VTKS[i]).alpha(
                0.1).linecolor("black").color("grey")
            rmesh.name = "ray_mesh"

        frames.append((a_meshes, p_meshes, fmesh, rmesh, info, tags))

        stats.append({
            "frame": i,
            "n_clusters": n_clusters,
            "n_noise_pts": n_noise,
            "n_polygons": n_poly,
            # "silhouette_coeff": s_coeff,
            "mean_area": np.nanmean(areas),
            "std_area": np.nanstd(areas),
            "mean_roundness": np.nanmean(roundnesses),
            "std_roundness": np.nanstd(roundnesses),
            "mean_max_diam": np.nanmean(max_diams),
            "std_max_diam": np.nanstd(max_diams),
        })
    stats_df = pd.DataFrame(stats)
    print("\nClustering and alpha shapes analysis complete.")
    print(stats_df)

    # plot the statistics over the simulation timeseries
    print("Generating shape statistics plots...")
    # figs = plot_alpha_shape_stats(stats_df)
    figs = plot_alpha_shape_stats_vedo(stats_df)
    print("Done")

    # render the output
    v = None  # video object
    video_length = 10
    if EXPORT:
        v = Video(
            name=f"../run/saves/{FOLDER_PATH.rsplit('/', maxsplit=1)[-1]}" +
            f"_{C_PROP}_f1.mp4",
            duration=video_length,
            backend="imageio")

    custom_shape = [  # position and size of rendering windows in plt
        dict(bottomleft=(0.00, 0.00), topright=(0.70, 1.00)),
        dict(bottomleft=(0.70, 0.00), topright=(1.00, 0.25)),
        dict(bottomleft=(0.70, 0.25), topright=(1.00, 0.50)),
        dict(bottomleft=(0.70, 0.50), topright=(1.00, 0.75)),
        dict(bottomleft=(0.70, 0.75), topright=(1.00, 1.00)),
    ]

    # p = Plotter(interactive=False, shape="1|4", size=(1920,1080),
    # sharecam=False)
    p = Plotter(interactive=False, shape=custom_shape,
                size=(1200, 900), sharecam=False)
    # p = Plotter(interactive=False, shape=(2,3), sharecam=False)
    # p.show(zoom="tight")#,axes=13)
    axes = Axes(xtitle="x", ytitle="y", ztitle="z",
                xrange=(max_bounds[0], max_bounds[1]),
                yrange=(max_bounds[2], max_bounds[3]))
    p.show(zoom="tight", axes=axes)
    # p.at(0).zoom(ZOOM)

    for frame, fig in zip(frames, figs):
        a_meshes, p_meshes, fmesh, rmesh, info, tags = frame
        fig1, fig2, fig3, fig4, fig5 = fig

        p.at(0).remove("a_mesh").add(a_meshes)
        p.at(0).remove("p_mesh").add(p_meshes)
        p.at(0).remove("fin_mesh").add(fmesh)
        p.at(0).remove("ray_mesh").add(rmesh)
        p.at(0).remove("info").add(info)
        p.at(0).remove("tag").add(tags)
        # p.at(1).remove("fig1").add(fig1).reset_camera(tight=0)
        p.at(2).remove("fig2").add(fig2).reset_camera(tight=0)
        p.at(3).remove("fig3").add(fig3).reset_camera(tight=0)
        p.at(4).remove("fig4").add(fig4).reset_camera(tight=0)
        p.at(1).remove("fig5").add(fig5).reset_camera(tight=0)
        p.render()
        if EXPORT:
            v.add_frame()
    if EXPORT:
        v.close()

    def slider1(widget, _):
        val = widget.value  # get the slider current value

        a_meshes, p_meshes, fmesh, rmesh, info, tags = frames[int(val)]

        p.at(0).remove("a_mesh").add(a_meshes)
        p.at(0).remove("p_mesh").add(p_meshes)
        p.at(0).remove("fin_mesh").add(fmesh)
        p.at(0).remove("ray_mesh").add(rmesh)
        p.at(0).remove("info").add(info)
        p.at(0).remove("tag").add(tags)

        if ANIMATE == 1:    # if matplotlib animation
            fig = figs[int(val)]
            p.at(1).remove("fig").add(fig).reset_camera(tight=0)

        if ANIMATE == 2:
            fig1, fig2, fig3, fig4, fig5 = figs[int(val)]
            # p.at(1).remove("fig1").add(fig1)
            p.at(2).remove("fig2").add(fig2)
            p.at(3).remove("fig3").add(fig3)
            p.at(4).remove("fig4").add(fig4)
            p.at(1).remove("fig5").add(fig5)

        p.render()

    p.at(0).add_slider(slider1, 0, len(frames)-1,
                       pos=([0.1, 0.9], [0.4, 0.9]), value=len(frames))
    p.interactive()  # .close()

    return stats_df


def plot_alpha_shape_stats_vedo(d):
    """Plot the spot shape statistics over simulation timecourse from
    dataframe of statistics using vedo.
    fig1 ... no. clusters
    fig2 ... no. polygons
    fig3 ... mean area
    fig4 ... mean roundness
    fig5 ... mean max diameter
    """

    t = 2.0  # text scale

    figs = []
    for i in range(len(d)):

        fig1 = pyplot.plot(np.array(d["frame"]), np.array(d["n_clusters"]),
                           xtitle="i", ytitle="No.clusters",
                           axes={"text_scale": t}, la=0.8)
        fig1 += Line(p0=(i, -1000, 0), p1=(i, 1000, 0), c="red")
        fig1.name = "fig1"

        fig2 = pyplot.plot(np.array(d["frame"]), np.array(d["n_polygons"]),
                           xtitle="i", ytitle="No. polygons",
                           axes={"text_scale": t}, la=0.8)
        fig2 += Line(p0=(i, -1000, 0), p1=(i, 1000, 0), c="red")
        fig2.name = "fig2"

        fig3 = pyplot.plot(np.array(d["frame"]), np.array(d["mean_area"]),
                           # add small number to prevent errors - careful
                           yerrors=np.array(d["std_area"])+0.0000001,
                           error_band=True, ec="grey",
                           xtitle="i", ytitle="Mean area",
                           axes={"text_scale": t}, xlim=(0, None),
                           la=0.8)
        fig3 += Line(p0=(i, -1000, 0), p1=(i, 1000, 0), c="red")
        fig3.name = "fig3"

        fig4 = pyplot.plot(np.array(d["frame"]),
                           np.array(d["mean_roundness"]),
                           yerrors=np.array(d["std_roundness"])+0.0000001,
                           error_band=True, ec="grey",
                           xtitle="i", ytitle="Mean roundness",
                           axes={"text_scale": t}, la=0.8)
        fig4 += Line(p0=(i, -1000, 0), p1=(i, 1000, 0), c="red")
        fig4.name = "fig4"

        fig5 = pyplot.plot(np.array(d["frame"]),
                           np.array(d["mean_max_diam"]),
                           yerrors=np.array(d["std_max_diam"])+0.0000001,
                           error_band=True, ec="grey",
                           xtitle="i", ytitle="Mean max diameter",
                           axes={"text_scale": t}, xlim=(0, None), la=0.8)
        fig5 += Line(p0=(i, -1000, 0), p1=(i, 1000, 0), c="red")
        fig5.name = "fig5"

        figs.append((fig1, fig2, fig3, fig4, fig5))

    return figs


def plot_alpha_shape_stats(d):
    """Plot the spot shape statistics over simulation timecourse from
    dataframe of statistics using matplotlib"""

    figs = []
    if ANIMATE == 1:
        for i in range(len(d)):
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=300)
            axs = axs.flatten()

            ax = axs[0]
            ax.plot(d["frame"], d["n_clusters"])
            ax.axvline(x=i, c="C1")
            ax.set_ylabel("No. clusters")

            ax = axs[1]
            ax.plot(d["frame"], d["n_polygons"])
            ax.axvline(x=i, c="C1")
            ax.set_ylabel("No. alpha polygons")

            ax = axs[2]
            ax.plot(d["frame"], d["mean_area"])
            ax.fill_between(d["frame"], d["mean_area"] - d["std_area"],
                            d["mean_area"] + d["std_area"], alpha=0.2)
            ax.axvline(x=i, c="C1")
            ax.set_ylim(0, None)
            ax.set_ylabel(r"Mean spot area $\pm$std")

            ax = axs[3]
            ax.plot(d["frame"], d["mean_roundness"])
            ax.fill_between(d["frame"], d["mean_roundness"] -
                            d["std_roundness"], d["mean_roundness"] +
                            d["std_roundness"], alpha=0.2)
            ax.axvline(x=i, c="C1")
            ax.set_ylim(0, None)
            ax.set_ylabel(r"Mean spot roundness $(\frac{4\pi \times "
                          r"\text{Area}}{\text{Perimeter}^2})$ $\pm$std")

            fig.supxlabel("Frame no.")
            fig.tight_layout()
            figs.append(fig)
            plt.close(fig)

        for i, fig in enumerate(figs):
            figs[i] = image.Image(fig)  # turn in into image

    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=300)
        axs = axs.flatten()

        ax = axs[0]
        ax.plot(d["frame"], d["n_clusters"])
        ax.set_ylabel("No. clusters")

        ax = axs[1]
        ax.plot(d["frame"], d["n_polygons"])
        ax.set_ylabel("No. alpha polygons")

        ax = axs[2]
        ax.plot(d["frame"], d["mean_area"])
        ax.fill_between(d["frame"], d["mean_area"] - d["std_area"],
                        d["mean_area"] + d["std_area"], alpha=0.2)
        ax.set_ylim(0, None)
        ax.set_ylabel(r"Mean spot area $\pm$std")

        ax = axs[3]
        ax.plot(d["frame"], d["mean_roundness"])
        ax.fill_between(d["frame"], d["mean_roundness"] - d["std_roundness"],
                        d["mean_roundness"] + d["std_roundness"], alpha=0.2)
        ax.set_ylim(0, None)
        ax.set_ylabel(r"Mean spot roundness $(\frac{4\pi \times "
                      r"\text{Area}}{\text{Perimeter}^2})$ $\pm$std")

        fig.supxlabel("Frame no.")
        fig.tight_layout()
        figs = image.Image(fig)  # turn into image
    # plt.show()

    return figs


def plot_alpha_shapes(uniq_labs, labels, core_samples_mask, spot_cells,
                      a_shapes):
    """Matplotlib function for plotting alpha shapes and clusters"""

    cs = [plt.cm.Spectral(i) for i in np.linspace(0, 1, len(uniq_labs))]
    _, ax = plt.subplots(figsize=(10, 9))
    for k, col in zip(uniq_labs, cs):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = spot_cells[class_member_mask & core_samples_mask]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=tuple(col),
            edgecolors="k",
        )

        xy = spot_cells[class_member_mask & ~core_samples_mask]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=tuple(col),
            edgecolors="k",
        )
        a_shape = a_shapes[k][1]
        print(type(a_shape))
        # if isinstance(a_shape, shapely.geometry.polygon.Polygon):
        if a_shape.geom_type == "Polygon":
            a_shape_poly = list(a_shape.exterior.coords)
            ax.add_patch(Polygon(a_shape_poly, facecolor=col, alpha=0.2))

    plt.title(f"Estimated number of clusters: {len(uniq_labs)}")
    plt.show()


def tissue_properties():
    """Return properties of the tissue over the entire timecourse, not
    deducible individual vtk files e.g. maximum no. cell types"""
    cell_types = []
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    pad = 0.1
    for vtk in VTKS:
        cell_types_i = np.unique(vtk.pointdata["cell_type"])
        if len(cell_types_i) > len(cell_types):
            cell_types = cell_types_i  # get the maximum no. cell types

        mesh = Mesh(vtk)
        if mesh.bounds()[0] < min_x:
            min_x = mesh.bounds()[0]
        if mesh.bounds()[1] > max_x:
            max_x = mesh.bounds()[1]
        if mesh.bounds()[2] < min_y:
            min_y = mesh.bounds()[2]
        if mesh.bounds()[3] > max_y:
            max_y = mesh.bounds()[3]

    max_bounds = (min_x - pad, max_x + pad, min_y - pad, max_y + pad)
    return cell_types, max_bounds


def get_vtks():
    """Return lists of cell and wall vtk files in the output folder"""

    cell = sorted(
        (f"{FOLDER_PATH}/{i}" for i in os.listdir(FOLDER_PATH)
         if i.endswith(".vtk") and "wall" not in i and "fin" not in i),
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1])
    )
    wall = sorted((f"{FOLDER_PATH}/{i}" for i in os.listdir(
        f"{FOLDER_PATH}") if i.endswith(".vtk") and "wall" in i),
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
    fin = sorted((f"{FOLDER_PATH}/{i}" for i in os.listdir(
        f"{FOLDER_PATH}") if i.endswith(".vtk") and "fin" in i and
        "ray" not in i),
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
    rays = sorted((f"{FOLDER_PATH}/{i}" for i in os.listdir(
        f"{FOLDER_PATH}") if i.endswith(".vtk") and "fin_ray" in i),
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))

    return cell, wall, fin, rays


def print_help():
    """Print help message for the script"""
    help_message = """
    Usage: python3 render.py [vtk_directory] [options]

    Options:
        -h              Show this help message and exit.
        -e              Export the rendering to video file.
        -c [c_prop]     Specify which cell property to colourise.
                        e.g. cell_type, u, mech_str
        -f [function]   Pass which function you want to perform:
                        0   ...render moive (default)
                        1   ...cluster spot cells and get spot stats
                        2   ...screenshot of last vtk file
        -z [zoom]       Zoom factor for camera view (0.6 is default)
        -w [walk id]    id of the walk being rendered
        -s [step]       Step of the walk being rendered
        -p [pt size]    Size of the points in the rendering
                        (default is 12, for z=0.3,p=26)
        """

    print(help_message)


if __name__ == "__main__":
    # collect bash arguments
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    else:
        output_folder = args[0].rstrip("/")  # remove trailing /
        # fetch custom parameters
        if "-e" in args:
            EXPORT = True
        if "-c" in args:
            C_PROP = str(args[args.index("-c") + 1])  # idx comes after -c flag
        if "-f" in args:
            FUNC = int(args[args.index("-f") + 1])
        if "-z" in args:
            ZOOM = float(args[args.index("-z") + 1])
        if "-w" in args:
            WALK_ID = int(args[args.index("-w") + 1])
        if "-s" in args:
            STEP = int(args[args.index("-s") + 1])
        if "-p" in args:
            PT_SIZE = int(args[args.index("-p") + 1])

        FOLDER_PATH = f'../run/{output_folder}'  # directory
        W_VTKS, F_VTKS, R_VTKS = None, None, None

        if FUNC == 0:
            cell_vtks, wall_vtks, fin_vtks, ray_vtks = get_vtks()
            VTKS = load(cell_vtks)
            if WALLS and len(wall_vtks) > 0:
                W_VTKS = load(wall_vtks)
            if FIN and len(fin_vtks) > 0:
                F_VTKS = load(fin_vtks)
            if RAYS and len(ray_vtks) > 0:
                R_VTKS = load(ray_vtks)
            render_movie(walls=WALLS and len(wall_vtks) > 0,
                         fin=FIN and len(fin_vtks) > 0,
                         rays=RAYS and len(ray_vtks) > 0,
                         cells=CELLS)
        elif FUNC == 1:
            cell_vtks, wall_vtks, fin_vtks, ray_vtks = get_vtks()
            if FIN and len(fin_vtks) > 0:
                F_VTKS = load(fin_vtks)
            if RAYS and len(ray_vtks) > 0:
                R_VTKS = load(ray_vtks)
            VTKS = load(cell_vtks)
            pattern_stats(fin=FIN, rays=RAYS)
        elif FUNC == 2:
            VTKS = load(f"{FOLDER_PATH}/out_{WALK_ID}_{STEP}_*.vtk")
            render_frame()
