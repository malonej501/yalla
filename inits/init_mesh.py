import sys
from vedo import *

DISPLAY = True
EXTRUDE = True
FORMAT = 0  # 0: legacy vtk yalla compatible, 1: new vtk (from vedo)
SHAPE = 2  # 0: 2D fin shape, 1: 2D rectangular shape,
EXTRUDE_Z = 0.1 if SHAPE != 2 else 100  # extrude distance in z direction

# 2: 2D rect with specific dimensions


def get_shape():
    """
    Returns the vertices and faces of the fin mesh based on the SHAPE 
    variable.
    """
    verts, faces = [], []
    if SHAPE == 0:
        # 2D fin shape
        verts = [(-1, 0.5, 0), (0.75, 0.5, 0), (0.75, -0.5, 0),
                 (0.2, -0.5, 0), (-1, 0.1, 0)]
        faces = [[0, 1, 2, 3, 4]]

    if SHAPE == 1:
        # 2D rectangular shape
        verts = [(-1, 0.5, 0), (1, 0.5, 0), (1, -0.5, 0), (-1, -0.5, 0)]
        faces = [[0, 1, 2, 3]]

    if SHAPE == 2:
        # 2D rectangular shape centered at (0,0,0) with specified dimensions
        ap = 7500  # anterior-posterior length
        pd = 2500  # proximal-distal height
        verts = [(-ap/2, pd/2, 0), (ap/2, pd/2, 0),
                 (ap/2, -pd/2, 0), (-ap/2, -pd/2, 0)]
        faces = [[0, 1, 2, 3]]

    return verts, faces


def export_vtk_custom(verts, faces):
    """Export verts and faces manually to legacy vtk"""

    # Create the vtk file
    with open(f"shape{SHAPE}_mesh_{'3D' if EXTRUDE else '2D'}.vtk",
              "w", encoding="utf-8") as vtk_file:

        # Write the header
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("vtk output\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        # Write the vertices
        vtk_file.write("POINTS " + str(len(verts)) + " float\n")
        for vert in verts:
            vtk_file.write(str(vert[0]) + " " +
                           str(vert[1]) + " " + str(vert[2]) + "\n")

        # Write the faces
        vtk_file.write("POLYGONS " + str(len(faces)) +
                       " " + str(len(faces) + 4) + "\n")
        for face in faces:
            vtk_file.write(str(len(face)))
            for vert in face:
                vtk_file.write(" " + str(vert))
            vtk_file.write("\n")


def extrude_mesh(verts):
    """Extrude 2D shape in the z direction by a specified distance mesh"""

    # Extrude in the z direction
    verts_3d = [(x, y, z + EXTRUDE_Z) for (x, y, z) in verts] + \
        [(x, y, z - EXTRUDE_Z) for (x, y, z) in verts]

    # Create faces for the extruded mesh
    num_verts = len(verts)
    faces_3d = []

    # Top and bottom faces
    faces_3d.append([i for i in range(num_verts)])
    faces_3d.append([i + num_verts for i in range(num_verts)])

    # Side faces
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
        faces_3d.append([i, next_i, next_i + num_verts, i + num_verts])

    return verts_3d, faces_3d


if __name__ == "__main__":

    # collect bash arguments
    args = sys.argv[1:]

    # print parameters
    print(f"DISPLAY: {DISPLAY}")
    print(f"EXTRUDE: {EXTRUDE}")
    print(f"EXTRUDE_Z: {EXTRUDE_Z}")
    print(f"FORMAT: {FORMAT}")
    print(f"SHAPE: {SHAPE}")

    v, f = get_shape()  # get the shape vertices and faces

    if EXTRUDE:
        v, f = extrude_mesh(v)  # extrude in z-direction

    # use vedo to triangulate the mesh - essential for yalla compatibility
    f_mesh = Mesh([v, f])
    f_mesh = f_mesh.triangulate()

    if DISPLAY:
        f_mesh.color("grey").linecolor("black")  # .alpha(0.5)
        plt = Plotter(axes=1)
        plt.add(f_mesh)
        plt.show(azimuth=-30, elevation=-10, interactive=False)
        plt.screenshot(f"shape{SHAPE}_mesh_{'3D' if EXTRUDE else '2D'}.png")
        plt.interactive()

    if FORMAT == 0:
        export_vtk_custom(f_mesh.vertices, f_mesh.cells)
    elif FORMAT == 1:
        f_mesh.write(f"shape{SHAPE}_mesh.vtk", binary=False)
