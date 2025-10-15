import sys
from vedo import *

DISPLAY = True
EXTRUDE = True
FORMAT = 0  # 0: legacy vtk yalla compatible, 1: new vtk (from vedo)
SHAPE = 4  # 0: 2D fin shape, 1: 2D rectangular shape, 3: 3D microscopy fin
EXTRUDE_Z = 0.1  # extrude distance in z direction

# 2: 2D rect with specific dimensions


def get_shape():
    """
    Returns the vertices and faces of the fin mesh based on the SHAPE 
    variable.
    """
    verts, faces, norms = [], [], []
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
        ap = 7.5  # anterior-posterior length mm
        pd = 2.5  # proximal-distal height mm
        verts = [(-ap/2, pd/2, 0), (ap/2, pd/2, 0),
                 (ap/2, -pd/2, 0), (-ap/2, -pd/2, 0)]
        faces = [[0, 1, 2, 3]]

    if SHAPE == 3:
        ap = 7.5  # anterior-posterior length mm
        pd = 2.5  # proximal-distal height mm
        verts = [(0, pd/2, 0),  # top
                 (0, -pd/2, 0),  # bottom
                 (-ap/2, 0, 0),  # left
                 (ap/2, 0, 0)]  # right
        norms = [(0, 1, 0),
                 (0, -1, 0),
                 (-1, 0, 0),
                 (1, 0, 0)]
        faces = [[0, 1, 2, 3, 4, 5, 6, 7]]
    if SHAPE == 4:  # fin shape with rays
        nrays = 11 + 2  # from counting images + 2 for the ends
        ap = 3  # ap length at proximal side
        height = 2  # control fin height
        maxd = 1.5  # max fin height
        smooth = 1  # more = sharper sin function
        # assume equal spacing along proximal side
        p_verts = [(0 + i*ap/(nrays-1), 0, 0) for i in range(nrays)]
        theta = 180 - 40  # degrees, angle of rays

        # def func(x):
        #     # quadratic fin edge
        #     return (maxd*4)/(nrays**2)*(x-(nrays/2))**2 - maxd
        def func(x):  # clipped sine fin edge
            return height * np.tanh(smooth * -np.sin(np.pi * (x / nrays)))
        f_lens = [func(i)for i in range(nrays)]
        d_verts = [(p_verts[i][0] + f_lens[i]*np.cos(np.radians(theta)),
                    f_lens[i]*np.sin(np.radians(theta)), 0) for i in range(nrays)]

        # start from the posterior proximal and go clockwise, leave
        # out the duplicate vertices at the ends
        verts = [p for p in reversed(p_verts)] + [d for d in d_verts[1:-1]]
        print(verts)
        faces = [[i for i in range(len(verts) + 2)]]

    return verts, faces, norms


def export_vtk_custom(verts, faces, norms):
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

        # Write the normals if they exist
        if len(norms) == len(verts):
            vtk_file.write("POINT_DATA " + str(len(verts)) + "\n")
            vtk_file.write("NORMALS polarity float\n")
            for norm in norms:
                vtk_file.write(str(norm[0]) + " " +
                               str(norm[1]) + " " + str(norm[2]) + "\n")


def extrude_mesh(verts, amount=EXTRUDE_Z):
    """Extrude 2D shape in the z direction by a specified distance mesh"""

    # Extrude in the z direction
    verts_3d = [(x, y, z + amount) for (x, y, z) in verts] + \
        [(x, y, z - amount) for (x, y, z) in verts]

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


def import_lmks(filename):
    """Import landmark points from a CSV file."""
    vtk = load("../data/lmk_DA-1-10_12-09-25/DA-1-10_12-07_0_lmk.vtk")

    v = vtk.vertices
    f = vtk.cells
    n = vtk.pointdata["polarity"]

    return v, f, n


if __name__ == "__main__":

    # collect bash arguments
    args = sys.argv[1:]

    # print parameters
    print(f"DISPLAY: {DISPLAY}")
    print(f"EXTRUDE: {EXTRUDE}")
    print(f"EXTRUDE_Z: {EXTRUDE_Z}")
    print(f"FORMAT: {FORMAT}")
    print(f"SHAPE: {SHAPE}")

    v, f, n = get_shape()  # get the shape vertices, faces and polarities
    # v, f, n = import_lmks(
    #     "../data/lmk_DA-1-10_12-09-25/DA-1-10_12-07_0_lmk.vtk")

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
        export_vtk_custom(f_mesh.vertices, f_mesh.cells, n)
    elif FORMAT == 1:
        f_mesh.write(f"shape{SHAPE}_mesh.vtk", binary=False)
