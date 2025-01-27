import sys
from vedo import *

display = True
extrude = True
extrude_z = 0.1 # extrude distance in z direction
format = 0 # 0: legacy vtk, 1: new vtk (from vedo)

# build 2D shape
verts = [(-1,0.5,0), (0.75,0.5,0), (0.75,-0.5,0), (0.2,-0.5,0), (-1,0.1,0)]
faces = [[0,1,2,3,4]]


def export_vtk_custom(verts, faces):
    """Export verts and faces manually to legacy vtk"""

    # Create the vtk file
    vtk_file = open("fin_mesh.vtk", "w")

    # Write the header
    vtk_file.write("# vtk DataFile Version 3.0\n")
    vtk_file.write("vtk output\n")
    vtk_file.write("ASCII\n")
    vtk_file.write("DATASET POLYDATA\n")

    # Write the vertices
    vtk_file.write("POINTS " + str(len(verts)) + " float\n")
    for vert in verts:
        vtk_file.write(str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\n")

    # Write the faces
    vtk_file.write("POLYGONS " + str(len(faces)) + " " + str(len(faces) + 4) + "\n")
    for face in faces:
        vtk_file.write(str(len(face)))
        for vert in face:
            vtk_file.write(" " + str(vert))
        vtk_file.write("\n")

    # Close the file
    vtk_file.close()

    return

def extrude_mesh(verts):
    """Extrude 2D shape in the z direction by a specified distance mesh"""

    # Extrude in the z direction by 0.1 units
    verts_3d = [(x, y, z + extrude_z) for (x, y, z) in verts] + [(x, y, z - extrude_z) for (x, y, z) in verts]

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

    # extrude in z-direction
    verts, faces = extrude_mesh(verts) if extrude else None
        
    # use vedo to triangulate the mesh - this is essential for loading into yalla
    f_mesh = Mesh([verts,faces])
    f_mesh = f_mesh.triangulate()

    if display:
        f_mesh.color("grey").linecolor("black").alpha(0.5)
        show(f_mesh, axes=1) 

    if format == 0:
        export_vtk_custom(f_mesh.points(), f_mesh.faces())
    elif format == 1:
        f_mesh.write("fin_mesh.vtk", binary=False)


