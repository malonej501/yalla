from vedo import *
import imageio
import os
import sys
import shapely
import math
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib import animation
import copy
import itertools
import pandas as pd
import numpy as np
import alphashape
from sklearn import metrics
from sklearn.cluster import DBSCAN

# Default parameters

# Visualisation
export = False # export the rendering to video file
c_prop = "cell_type" # cell property to colourise
func = 0 # by default render movie
zoom = 0.6 # define the how far the camera is out
pt_size = 12 # how large the cells are drawn
animate = 2 # 0 = False, 1 = Matplotlib, 2 = Vedo
ax = True # show axes

# DBSCAN parameters
eps = 0.05 # maximum distance between two samples for one to be considered as in the neighborhood of the other

def render_movie(vtks, folder_path):

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
    plt.show(zoom="tight", axes=1 if ax == True else 0) #zoom="tight")#,axes=ax)#, axes=13)
    plt.zoom(zoom)

    if export:
        v = Video(
            name=f"{folder_path.split('/')[-1]}_{c_prop}.mp4", 
            duration=video_length, 
            backend="imageio")
    
    frames = []
    # Loop through the VTK files and visualize them
    for i, vtk in enumerate(vtks):

        points = Points(vtk).point_size(pt_size * zoom) #originally 10
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
       
        plt.render()#.reset_camera()
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

def pattern_stats(vtks, folder_path):
    
    """Cluster spot cells and infer alpha shapes and shape stats"""
    stats = []
    frames = []
    print(f"Clustering and alpha shapes analysis: {folder_path}")
    for i, mesh in enumerate(vtks):
        sys.stdout.write(f"\rFrame: {i}")
        sys.stdout.flush()
        mesh = vtks[i]
        X_spots = mesh.vertices[mesh.pointdata["cell_type"] == 1] # return positions of spot cells
        if X_spots.size == 0: # skip the iteration if there are no spot cells present in the vtk
            continue
        # eps - maximum distance between two samples for one to be considered as in the neighborhood of the other - set to equilibrium distance between spot cells - check force potential
        db = DBSCAN(eps=eps, min_samples=1).fit(X_spots)
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
        n_poly = 0
        tags = []
        a_meshes = [] # the alpha shapes
        p_meshes = [] # the spot cell points
        # return the coordinates of the cells in each cluster
        for l in unique_labels:
            class_member_mask = labels == l
            xy = X_spots[class_member_mask & core_samples_mask]
            p_mesh = Points(xy).point_size(pt_size * zoom).color(l)
            p_mesh.name = "p_mesh"
            p_meshes.append(p_mesh)
            xy = np.delete(xy, 2, axis=1) # remove z dimension
            alpha_shape = alphashape.alphashape(xy, alpha=15)

            # plot alpha shape with Polygon(list(alpha_shape.exterior.coords), alpha=0.2)
            a_shapes.append((l, alpha_shape))
            areas.append(alpha_shape.area) # calculate area for all types of geometries, not just polygons
            if alpha_shape.geom_type == "Polygon": #or alpha_shape.geom_type == "MultiPolygon":
                n_poly += 1 # count number of polygons
                a_shape_np = np.array(alpha_shape.exterior.coords)
                z_col = np.zeros((a_shape_np.shape[0],1))
                a_shape_np = np.hstack((a_shape_np, z_col))
                cells = np.array([range(len(a_shape_np))])

                a_mesh = Mesh([a_shape_np,cells])
                a_mesh_ids = a_mesh.labels(content=np.array([l]), on="cells",yrot=180) # only display label if polygon
                a_mesh_ids.name = "tag"
                tags.append(a_mesh_ids)
                a_mesh.color(l).alpha(0.2)
                a_mesh.triangulate() # to ensure neat rendering of final shapes
                a_mesh.name = "a_mesh"
                a_meshes.append(a_mesh)

                perimeter = shapely.length(alpha_shape)
                roundnesses.append((4 * math.pi * alpha_shape.area) / (perimeter**2)) # 1 for perfect circle, 0 for non-circular
            if alpha_shape.geom_type == "LineString": 
                a_meshes.append(None)
                tags.append(None)
                roundnesses.append(0) # if line roundness is 0
            if alpha_shape.geom_type == "Point": 
                a_meshes.append(None)
                tags.append(None)
                roundnesses.append(1) # if point roundness is 1
                
        info = Text2D(
            txt=(f"i: {i}\n"
            f"n_clusters: {len(unique_labels)}\n"
            f"n_polygons: {n_poly}\n"
            f"mean_area: {np.mean(areas):.4f}\n"
            f"std_area: {np.std(areas):.4f}\n"
            f"mean_roundnesses: {np.mean(roundnesses):.4f}\n"
            f"std_roundnesses: {np.std(roundnesses):.4f}\n"),
            pos="bottom-left")
        info.name = "info"

        frames.append((a_meshes,p_meshes,info,tags))
        
        stats.append({
            "frame": i,
            "n_clusters": n_clusters,
            "n_noise_pts": n_noise,
            "n_polygons": n_poly,
            # "silhouette_coeff": s_coeff,
            "mean_area": np.mean(areas),
            "std_area": np.std(areas),
            "mean_roundness": np.mean(roundnesses),
            "std_roundness": np.std(roundnesses)
        })
    stats_df = pd.DataFrame(stats)
    print("\nClustering and alpha shapes analysis complete.")  
 
    # plot the statistics over the simulation timeseries
    print("Generating shape statistics plots...")
    # figs = plot_alpha_shape_stats(stats_df)
    figs = plot_alpha_shape_stats_vedo(stats_df)
    print("Done")

    # render the output
    video_length=10
    if export:
        v = Video(
            name=f"{folder_path.split('/')[-1]}_{c_prop}_f1.mp4", 
            duration=video_length, 
            backend="imageio")

    plt = Plotter(interactive=False, shape="1|4", size=(1920,1080), sharecam=False)
    # plt = Plotter(interactive=False, shape=(2,3), sharecam=False)
    # plt.show(zoom="tight")#,axes=13)
    plt.show(zoom="tight",axes=1)
    plt.at(0).zoom(zoom)

    for frame, fig in zip(frames, figs):
        a_meshes, p_meshes, info, tags = frame
        fig1, fig2, fig3, fig4 = fig

        plt.at(0).remove("a_mesh").add(a_meshes)
        plt.at(0).remove("p_mesh").add(p_meshes)
        plt.at(0).remove("info").add(info)
        plt.at(0).remove("tag").add(tags)
        plt.at(1).remove("fig1").add(fig1).reset_camera(tight=0)
        plt.at(2).remove("fig2").add(fig2).reset_camera(tight=0)
        plt.at(3).remove("fig3").add(fig3).reset_camera(tight=0)
        plt.at(4).remove("fig4").add(fig4).reset_camera(tight=0)
        plt.render()

        v.add_frame() if export else None

    v.close() if export else None

    def slider1(widget, event):
        val = widget.value # get the slider current value

        a_meshes, p_meshes, info, tags= frames[int(val)]

        plt.at(0).remove("a_mesh").add(a_meshes)
        plt.at(0).remove("p_mesh").add(p_meshes)
        plt.at(0).remove("info").add(info)
        plt.at(0).remove("tag").add(tags)

        if animate == 1:    # if matplotlib animation
            fig = figs[int(val)]
            plt.at(1).remove("fig").add(fig).reset_camera(tight=0)
        
        if animate == 2:
            fig1, fig2, fig3, fig4 = figs[int(val)]
            plt.at(1).remove("fig1").add(fig1)
            plt.at(2).remove("fig2").add(fig2)
            plt.at(3).remove("fig3").add(fig3)
            plt.at(4).remove("fig4").add(fig4)
            
        plt.render()

    plt.at(0).add_slider(slider1, 0, len(frames)-1, pos=([0.1,0.9],[0.4,0.9]), value=len(frames))
    plt.interactive()#.close()

    return stats_df

def plot_alpha_shape_stats_vedo(d):
    """Plot the spot shape statistics over simulation timecourse from dataframe of statistics"""
    figs = []
    print(d)
    # d.fillna(0, inplace=True)
    for i in range(len(d)):
        
        fig1 = pyplot.plot(np.array(d["frame"]),np.array(d["n_clusters"]),
        xtitle="i", ytitle="No.clusters")
        fig1 += Line(p0=(i,-1000,0),p1=(i,1000,0),c="red")
        fig1.name = "fig1"

        fig2 = pyplot.plot(np.array(d["frame"]),np.array(d["n_polygons"]),
        xtitle="i", ytitle="No. polygons")
        fig2 += Line(p0=(i,-1000,0),p1=(i,1000,0),c="red")
        fig2.name = "fig2"
        
        fig3 = pyplot.plot(np.array(d["frame"]),np.array(d["mean_area"]),
        yerrors=np.array(d["std_area"])+0.0000001, error_band=True, ec="grey", # add small number to prevent errors - may lead to bugs later so careful
        xtitle="i", ytitle="Mean area")
        fig3 += Line(p0=(i,-1000,0),p1=(i,1000,0),c="red")
        fig3.name = "fig3"

        fig4 = pyplot.plot(np.array(d["frame"]),np.array(d["mean_roundness"]),
        yerrors=np.array(d["std_roundness"])+0.0000001, error_band=True, ec="grey",
        xtitle="i", ytitle="Mean roundness")
        fig4 += Line(p0=(i,-1000,0),p1=(i,1000,0),c="red")
        fig4.name = "fig4"
    
        figs.append((fig1, fig2, fig3, fig4))

    return figs

def plot_alpha_shape_stats(d):

    figs = []
    if animate == 1:
        for i in range(len(d)):
            fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(10,8),dpi=300)
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
            ax.fill_between(d["frame"], d["mean_area"] - d["std_area"], d["mean_area"] + d["std_area"], alpha=0.2)
            ax.axvline(x=i, c="C1")
            ax.set_ylim(0,None)
            ax.set_ylabel(r"Mean spot area $\pm$std")

            ax = axs[3]
            ax.plot(d["frame"], d["mean_roundness"])
            ax.fill_between(d["frame"], d["mean_roundness"] - d["std_roundness"], d["mean_roundness"] + d["std_roundness"], alpha=0.2)
            ax.axvline(x=i, c="C1")
            ax.set_ylim(0,None)
            ax.set_ylabel(r"Mean spot roundness $(\frac{4\pi \times \text{Area}}{\text{Perimeter}^2})$ $\pm$std")

            fig.supxlabel("Frame no.")
            fig.tight_layout()
            figs.append(fig)
            plt.close(fig)
        
        for i, fig in enumerate(figs):
            figs[i] = image.Image(fig) # turn in into image

    else:
        fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(10,8),dpi=300)
        axs = axs.flatten()
        
        ax = axs[0]
        ax.plot(d["frame"], d["n_clusters"])
        ax.set_ylabel("No. clusters")

        ax = axs[1]
        ax.plot(d["frame"], d["n_polygons"])
        ax.set_ylabel("No. alpha polygons")

        ax = axs[2]
        ax.plot(d["frame"], d["mean_area"])
        ax.fill_between(d["frame"], d["mean_area"] - d["std_area"], d["mean_area"] + d["std_area"], alpha=0.2)
        ax.set_ylim(0,None)
        ax.set_ylabel(r"Mean spot area $\pm$std")

        ax = axs[3]
        ax.plot(d["frame"], d["mean_roundness"])
        ax.fill_between(d["frame"], d["mean_roundness"] - d["std_roundness"], d["mean_roundness"] + d["std_roundness"], alpha=0.2)
        ax.set_ylim(0,None)
        ax.set_ylabel(r"Mean spot roundness $(\frac{4\pi \times \text{Area}}{\text{Perimeter}^2})$ $\pm$std")

        fig.supxlabel("Frame no.")
        fig.tight_layout()
        figs = image.Image(fig) # turn into image
    # plt.show()

    return figs


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
        -c [c_prop]     Specify which cell property to colourise.
                        e.g. cell_type, u, mech_str
        -f [function]   Pass which function you want to perform:
                        0   ...render moive (default)
                        1   ...cluster spot cells and get spot stats
        -z [zoom]       Zoom factor for camera view (0.6 is default)      
        """
    
    print(help_message)

    

if __name__ == "__main__":
    # collect bash arguments
    args = sys.argv[1:]
    if "-h" in args:
        print_help()
    else:
        output_folder = args[0]
        # fetch custom parameters
        if "-e" in args:
            export = True
        if "-c" in args:
            c_prop = str(args[args.index("-c") + 1]) # idx comes after -c flag
        if "-f" in args:
            func = int(args[args.index("-f") + 1])
        if "-z" in args:
            zoom = float(args[args.index("-z") + 1])
            

        folder_path = f'/home/jmalone/GitHub/yalla/run/saves/{output_folder}' # directory

        vtks = load(f"{folder_path}/*.vtk")
        if func == 0:
            render_movie(vtks, folder_path)
        elif func == 1:
            pattern_stats(vtks, folder_path)



