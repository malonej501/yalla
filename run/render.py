from vedo import *
import imageio
import os
import sys

def load_vtks(folder_path):
    
    print(f"Rendering: {folder_path}")

    # List all VTK files in the folder
    vtks = [f for f in os.listdir(folder_path) if f.endswith('.vtk')]

    # Sort files if needed
    vtks = sorted(vtks, key=lambda x: int(x.split('_')[1].split('.')[0]))

    frames = []
    for vtk in vtks:
        print(vtk)
        # Construct the full file path
        file_path = os.path.join(folder_path, vtk)

        # Read the VTK file
        frame = load(file_path)
        frames.append(frame)

    return frames

def render_movie(folder_path, export):
    
    video_length = 10 # in seconds
    cmap = "Set1" # the colour map
    c_prop = "cell_type" # which property of cells do you want to colour

    # Create a plotter
    plt = Plotter(interactive=0)
    plt.show(zoom="tight")

    if export == "-e":
        v = Video(name=f"{folder_path.split('/')[-1]}.mp4", duration=10, backend="imageio")

    # Load frames
    vtks = load_vtks(folder_path)

    # Loop through the VTK files and visualize them
    for vtk in vtks:

        points = Points(vtk).point_size(10)
        points.cmap("Set1","cell_type")
        #trace.c(trace.pointdata["cell_type"])
        #print(trace)
        # print(trace.pointdata["cell_type"])
        # exit()

        points.name = "cells"
        # Add the mesh to the plotter
        plt.remove("cells").add(points)

        plt.render().reset_camera()
        if export == "-e":
            v.add_frame()

        #plt.clear()
        #plt.remove().render()
        #time.sleep(1)

    if export == "-e":
        v.close()
    plt.interactive().close()
    plt.clear()

def show_chem_grad(folder_path):
    pts = load_vtks(folder_path)

    u = pts.pointdata["u"]
    print(u)

if __name__ == "__main__":
    # collect bash arguments
    args = sys.argv[1:]
    output_folder = args[0]
    export = False
    if len(args) == 2:
        export = args[1]

    folder_path = f'/home/jmalone/GitHub/yalla/run/saves/{output_folder}' # directory

    #render_movie(folder_path, export)
    show_chem_grad(folder_path)


