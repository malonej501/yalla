from vedo import *
import imageio
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

def load_vtks(folder_path):
    
    print(f"Rendering: {folder_path}")

    # List all VTK files in the folder
    vtks = [f for f in os.listdir(folder_path) if f.endswith('.vtk')]

    # Sort files if needed
    vtks = sorted(vtks, key=lambda x: int(x.split('_')[1].split('.')[0]))

    frames = []
    for vtk in vtks:
        # print(vtk)
        # Construct the full file path
        file_path = os.path.join(folder_path, vtk)

        # Read the VTK file
        frame = load(file_path)
        frames.append(frame)

    return frames

def render_movie(c_prop, folder_path, export):

    """Choose cell property to colourise - e.g. cell_type, u, mechanical_strain"""
    
    video_length = 10 # in seconds
    # cmap = "Set1" # the colour map
    #c_prop = "cell_type" # which property of cells do you want to colour
    cmap = "viridis"
    # c_prop = "u"

    # Create a plotter
    plt = Plotter(interactive=0)
    plt.show(zoom="tight")

    if export:
        v = Video(name=f"{folder_path.split('/')[-1]}_{c_prop}.mp4", duration=video_length, backend="imageio")
    # Load frames
    vtks = load_vtks(folder_path)

    # Loop through the VTK files and visualize them
    for vtk in vtks:

        points = Points(vtk).point_size(7) #originally 10
        points.cmap(cmap, c_prop)
        points.add_scalarbar(title=c_prop)
        # points.rotate_x(-45).rotate_y(-45)
        # points.lighting("plastic")
        # p1 = Point([2,2,2], c="white")
        # l1 = Light(p1, c="white")
        #trace.c(trace.pointdata["cell_type"])
        #print(trace)
        # print(trace.pointdata["cell_type"])
        # exit()

        points.name = "cells"
        # Add the mesh to the plotter
        plt.remove("cells").add(points)

        # points

        plt.render().reset_camera()
        if export:
            v.add_frame()

        #plt.clear()
        #plt.remove().render()
        #time.sleep(1)

    if export:
        v.close()
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

def tissue_stats(folder_path):
    pts = load_vtks(folder_path)

    stats = []
    for pt in pts:
        #print(pt.vertices[:10])
        xmax = max(pt.vertices[:, 0])
        ymax = max(pt.vertices[:, 1])
        xmin = min(pt.vertices[:, 0])
        ymin = min(pt.vertices[:, 1])

        n_A = len(pt.pointdata["cell_type"][pt.pointdata["cell_type"] == 1])
        n_B = len(pt.pointdata["cell_type"][pt.pointdata["cell_type"] == 2])
        n_dead = len(pt.pointdata["cell_type"][pt.pointdata["cell_type"] == 0])
        n_staging1 = len(pt.pointdata["cell_type"][pt.pointdata["cell_type"] == -1]) 
        n_staging2 = len(pt.pointdata["cell_type"][pt.pointdata["cell_type"] == -2])

        stats.append({
            'xmax': xmax,
            'ymax': ymax,
            'xmin': xmin,
            'ymin': ymin,
            'n_A': n_A,
            'n_B': n_B,
            'n_dead': n_dead,
            "n_staging1": n_staging1,
            "n_staging2": n_staging2
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df)

def print_help():
    help_message = """
    Usage: python3 render.py [options]
    
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

        tissue_stats(folder_path)
        render_movie(c_prop, folder_path, export)
        #show_chem_grad(folder_path)


