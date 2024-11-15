from vedo import *
import imageio
import os
import sys
import shapely
import math
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import alphashape
from sklearn import metrics
from sklearn.cluster import DBSCAN

zoom = 0.7 # define the how far the camera is out

def render_movie(c_prop, folder_path, export, vtks):

    """Choose cell property to colourise - e.g. cell_type, u, mechanical_strain"""

    print(f"Rendering: {folder_path}")
    
    video_length = 10 # in seconds
    cmap = "viridis"
    lims = ((-1,6),(-1,6))
    # lims = ((-10,10),(-10,10))

    # Create a plotter
    plt = Plotter(interactive=False)
    # ax = addons.Axes(plt, xrange=(lims[0][0],lims[0][1]), yrange=(lims[1][0],lims[1][1]), zrange=(0,0))
    # ax.name = "ax"
    plt.show(zoom="tight")#, axes=13)
    # plt.zoom(zoom)

    if export:
        v = Video(
            name=f"{folder_path.split('/')[-1]}_{c_prop}.mp4", 
            duration=video_length, 
            backend="imageio")
    
    frames = []
    # Loop through the VTK files and visualize them
    for i, vtk in enumerate(vtks):

        points = Points(vtk).point_size(7) #originally 10
        # lims = ((points.bounds()[0],points.bounds()[1]),(points.bounds()[2],points.bounds()[3]))

        points.cmap(cmap, c_prop)
        bar = addons.ScalarBar(points, title=c_prop)
        info = Text2D(
            txt=(f"i: {i}\n"
            f"n: {len(points.vertices)}\n"
            f"n_1: {len(points.pointdata["cell_type"][points.pointdata["cell_type"] == 1])}\n"
            f"n_2: {len(points.pointdata["cell_type"][points.pointdata["cell_type"] == 2])}\n"),
            pos="bottom-left")

        points.name = "cells"
        bar.name = "bar"
        info.name = "info"
        frames.append((points, bar, info))
        # Add the mesh to the plotter
        plt.remove("cells")
        plt.remove("bar")
        plt.remove("info")
        plt.add(points)
        plt.add(info)
        plt.add(bar)
       
        plt.render().reset_camera()
        if export:
            v.add_frame()

    if export:
        v.close()

    def slider1(widget, event):
        val = widget.value # get the slider current value

        plt.remove("cells")
        plt.remove("bar")
        plt.remove("info")

        points, bar, info = frames[int(val)]

        plt.add(points)
        plt.add(bar)
        plt.add(info)
        plt.render()

    def slider2(widget, event):
        val = int(widget.value)
        c_prop = points.pointdata.keys()[val]

        plt.remove("bar") # remove the old bar

        points.cmap(cmap, c_prop) # change the cmap of the current view
        bar = addons.ScalarBar(points, title=c_prop) # create the new bar
        bar.name = "bar"
     
        # change the cmap of all frames
        for k, (pts, br, info) in enumerate(frames):
            pts = pts.cmap(cmap, c_prop)
            br = addons.ScalarBar(pts, title=c_prop)
            br.name = "bar"
            frames[k] = (pts, br, info)
    
        plt.add(bar)
        plt.render()

    plt.add_slider(slider1, 0, len(frames)-1, pos="top-right", value=len(frames))
    plt.add_slider(slider2, 0, len(points.pointdata.keys())-1, pos="top-left", value=points.pointdata.keys().index(c_prop))
    plt.interactive().close()
    plt.clear()

def show_chem_grad(folder_path):
    pts = load_vtks(folder_path)

    n_t = 10

    print(pts[0])
    print(pts[0].pointdata["cell_type"])
    print(pts[0].vertices)
    print(pts[0].pointdata["u"])
    x_pos = pts[0].vertices[:, 0]

    u = pts[0].pointdata["u"]
    v = pts[0].pointdata["v"]
    attributes = [u,v]


    fig, axs = plt.subplots(n_t,2, figsize=(10,2*n_t))

    for i, row in enumerate(axs):
        t = int(i* len(pts)/n_t)
        u = pts[t].pointdata["u"]
        v = pts[t].pointdata["v"]
        attributes = [u,v]
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

def tissue_stats(vtks):

    stats = []
    for pt in vtks:
        #print(pt.vertices[:10])
        xmax = max(pt.vertices[:, 0])
        ymax = max(pt.vertices[:, 1])
        xmin = min(pt.vertices[:, 0])
        ymin = min(pt.vertices[:, 1])

        n_A = len(pt.pointdata["cell_type"][pt.pointdata["cell_type"] == 1])
        n_B = len(pt.pointdata["cell_type"][pt.pointdata["cell_type"] == 2])

        stats.append({
            'xmax': xmax,
            'ymax': ymax,
            'xmin': xmin,
            'ymin': ymin,
            'n_A': n_A,
            'n_B': n_B,
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df)

def pattern_stats(vtks):
    
    plt = Plotter(interactive=False)
    plt.show(zoom="tight")
    stats = []
    frames = []
    a_meshes = []
    p_meshes = []
    for i, mesh in enumerate(vtks):
        mesh = vtks[i]
        X_spots = mesh.vertices[mesh.pointdata["cell_type"] == 1] # return positions of spot cells
        # eps - maximum distance between two samples for one to be considered as in the neighborhood of the other - set to r_max
        db = DBSCAN(eps=0.1, min_samples=1).fit(X_spots)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1) # noisy points are given the label -1
        #https://scikit-learn.org/1.5/modules/clustering.html#silhouette-coefficient
        # s_coeff = metrics.silhouette_score(spot_cells, labels) # a higher Silhouette Coefficient score relates to a model with better defined clusters
        
        core_samples_mask = np.zeros_like(labels,dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        a_shapes = []
        areas = []
        roundnesses = [] # roundness is only computed for polygons - not lines, points etc.
        tags = []
        a_meshes = []
        p_meshes = []
        n_poly = 0
        # return the coordinates of the cells in each cluster
        for l in unique_labels:
            class_member_mask = labels == l
            xy = X_spots[class_member_mask & core_samples_mask]
            p_mesh = Points(xy).point_size(7).color(l)
            p_mesh.name = "p_mesh"
            p_meshes.append(p_mesh)
            xy = np.delete(xy, 2, axis=1) # remove z dimension
            alpha_shape = alphashape.alphashape(xy, alpha=10)

            # plot alpha shape with Polygon(list(alpha_shape.exterior.coords), alpha=0.2)
            a_shapes.append((l, alpha_shape))
            areas.append(alpha_shape.area)
            if alpha_shape.geom_type == "Polygon": #or alpha_shape.geom_type == "MultiPolygon":
                n_poly += 1 # count number of polygons
                a_shape_np = np.array(alpha_shape.exterior.coords)
                z_col = np.zeros((a_shape_np.shape[0],1))
                a_shape_np = np.hstack((a_shape_np, z_col))
                cells = np.array([range(len(a_shape_np))])

                a_mesh = Mesh([a_shape_np,cells])
                a_mesh_ids = a_mesh.labels(content=np.array([l]), on="cells",yrot=180)
                a_mesh_ids.name = "tag"
                tags.append(a_mesh_ids)
                a_mesh.color(l).alpha(0.2)
                a_mesh.name = "a_mesh"
                a_meshes.append(a_mesh)

                perimeter = shapely.length(alpha_shape)
                roundnesses.append((4 * math.pi * alpha_shape.area) / (perimeter**2)) # 1 for perfect circle, 0 for non-circular

        info = Text2D(
            txt=(f"i: {i}\n"
            f"n_clusters: {len(unique_labels)}\n"
            f"n_polygons: {n_poly}\n"
            f"mean_area: {np.mean(areas):.4f}\n"
            f"std_area: {np.std(areas):.4f}\n"
            f"mean_roundnesses: {np.mean(roundnesses):.4f}\n"),
            pos="bottom-left")
        info.name = "info"

        plt.remove("a_mesh")
        plt.remove("p_mesh")
        plt.remove("info")
        plt.remove("tag")
        frames.append((a_meshes,p_meshes,info,tags))
        plt.add(a_meshes)
        plt.add(p_meshes)
        plt.add(info)
        plt.add(tags)
        plt.render().reset_camera()

        # plot_alpha_shapes(unique_labels, labels, core_samples_mask, spot_cells, a_shapes)
        
        stats.append({
            "frame": i,
            "n_clusters": n_clusters,
            "n_noise_pts": n_noise,
            "n_polygons": n_poly,
            # "silhouette_coeff": s_coeff,
            "mean_area": np.mean(areas),
            "std_area": np.std(areas),
            "mean_roundness": np.mean(roundnesses)
        })

    stats_df = pd.DataFrame(stats)
    print(stats_df)

    def slider1(widget, event):
        val = widget.value # get the slider current value

        plt.remove("a_mesh")
        plt.remove("p_mesh")
        plt.remove("info")
        plt.remove("tag")

        a_meshes, p_meshes, info, tags= frames[int(val)]

        plt.add(a_meshes)
        plt.add(p_meshes)
        plt.add(info)
        plt.add(tags)
        plt.render()

    plt.add_slider(slider1, 0, len(frames)-1, pos="top-right", value=len(frames))
    plt.interactive().close()

    

    return stats_df


def plot_alpha_shapes(unique_labels, labels, core_samples_mask, spot_cells, a_shapes):

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    fig, ax = plt.subplots(figsize=(10,9))
    for k, col in zip(unique_labels, colors):
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
        #if isinstance(a_shape, shapely.geometry.polygon.Polygon):
        if a_shape.geom_type == "Polygon":
            a_shape_poly = list(a_shape.exterior.coords)
            ax.add_patch(Polygon(a_shape_poly, facecolor=col, alpha=0.2))
    

    plt.title(f"Estimated number of clusters: {len(unique_labels)}")
    plt.show()

# def render_alpha_shape_mesh(vtks):

#     _, a_shapes_np = pattern_stats(vtks)

#     # plt = applications.Browser(vtks, bg = 'k')
#     # plt.show(interactive = True).close()
#     print(mesh)
    

def print_help():
    help_message = """
    Usage: python3 render.py [vtk_directory] [options]
    
    Options:
        -h              Show this help message and exit.
        -e              Export the rendering to video file.
        -c c_prop       Specify which cell property to colourise.
        """
    
    print(help_message)

    

if __name__ == "__main__":
    # collect bash arguments
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    else:
        output_folder = args[0]
        export = False
        if "-e" in args:
            export = True
        if "-c" in args:
            c_prop_idx = args.index("-c") + 1 # identify the index of the c_prop argument - comes after -c flag
            c_prop = str(args[c_prop_idx])
        else:
            c_prop = "cell_type"

        folder_path = f'/home/jmalone/GitHub/yalla/run/saves/{output_folder}' # directory

        vtks = load(f"{folder_path}/*.vtk")
        # render_alpha_shape_mesh(vtks)
        pattern_stats(vtks)
        # tissue_stats(vtks)
        # render_movie(c_prop, folder_path, export, vtks)
        #show_chem_grad(folder_path)


