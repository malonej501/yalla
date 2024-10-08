from vedo import *
import os

# Specify the folder containing the VTK files
folder_path = 'saves/volk_2D_short_range_overcrowding_08-10-24'

# List all VTK files in the folder
vtks = [f for f in os.listdir(folder_path) if f.endswith('.vtk')]

print(vtks)
exit()
# Sort files if needed
vtk_files.sort()

# Create a plotter
plotter = Plotter()

# Loop through the VTK files and visualize them
for vtk_file in vtk_files:
    # Construct the full file path
    file_path = os.path.join(folder_path, vtk_file)

    # Read the VTK file
    mesh = load(file_path)

    # Add the mesh to the plotter
    plotter += mesh

    # Show the current frame and wait for a key press
    plotter.show(interactive=True)

# Close the plotter when done
plotter.close()
