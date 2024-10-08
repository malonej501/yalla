from vedo import *
import imageio
import os

folder_path = 'saves/volk_2D_short_range_overcrowding_08-10-24' # directory
video_length = 10 # in seconds
cmap = "Set1" # the colour map
c_prop = "cell_type" # which property of cells do you want to colour

# List all VTK files in the folder
vtks = [f for f in os.listdir(folder_path) if f.endswith('.vtk')]

# Sort files if needed
vtks = sorted(vtks, key=lambda x: int(x.split('_')[1].split('.')[0]))
print(vtks)

# Create a plotter
plt = Plotter(interactive=0)
plt.show(zoom="tight")
v=Video(name=f"{folder_path.split('/')[-1]}.mp4", duration=10, backend="imageio")

# Loop through the VTK files and visualize them
for vtk in vtks:
    print(vtk)
    # Construct the full file path
    file_path = os.path.join(folder_path, vtk)

    # Read the VTK file
    cells = load(file_path)


    points = Points(cells).point_size(10)
    points.cmap("Set1","cell_type")
    #trace.c(trace.pointdata["cell_type"])
    #print(trace)
    # print(trace.pointdata["cell_type"])
    # exit()

    points.name = "cells"
    # Add the mesh to the plotter
    plt.remove("cells").add(points)

    plt.render().reset_camera()
    v.add_frame()

    #plt.clear()
    #plt.remove().render()
    #time.sleep(1)

v.close()
plt.interactive().close()
plt.clear()
