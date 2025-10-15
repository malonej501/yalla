import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from vedo import *


class Fin:
    """Class to generate a fin shape with rays."""

    def __init__(self, nrays=11, ap=3, height=2, smooth=1, theta=140):
        self.nrays = nrays
        self.ray_width = 0.05  # width of rays
        self.ap = ap  # anterior-posterior length at proximal side
        self.height = height  # control fin height
        self.smooth = smooth  # smoothness factor for fin edge
        self.theta = theta  # angle of rays
        self.p_verts = []  # proximal vertices
        self.d_verts = []  # distal vertices
        self.pflat = []  # flattened proximal vertices
        self.dflat = []  # flattened distal vertices
        self.faces = []
        self.verts_3d = []
        self.rays = []
        self.ray_faces = []

        # Proximal ray midpoints
        p_verts = [[0 + i * ap / ((nrays+2) - 1), 0, 0]
                   for i in range(nrays+2)]
        # Proximal vertices for start and end of fin rays
        # Build proximal vertices with ray width offsets
        p_verts_new = [[p_verts[0]]]
        for v in p_verts[1:-1]:
            p_pair = [[v[0] + self.ray_width/2, 0, 0],  # assume order p->a
                      [v[0] - self.ray_width/2, 0, 0]]
            p_verts_new.append(p_pair)
        p_verts_new.append([p_verts[-1]])
        p_verts = list(reversed(p_verts_new))  # order p->a

        # Fin edge function
        def func(x):
            return height * np.tanh(smooth * -np.sin(np.pi * (x / self.ap)))

        # Distal vertices
        d_verts = []
        for pair in p_verts:
            d_pair = []
            for v in pair:
                d_v = [float(v[0] + func(v[0]) * np.cos(np.radians(theta))),
                       float(func(v[0]) * np.sin(np.radians(theta))), 0]
                d_pair.append(d_v)
            d_pair = list(reversed(d_pair))  # order a->p
            d_verts.append(d_pair)
        d_verts = list(reversed(d_verts))  # order a->p

        # Remove endpoints
        self.p_verts = list(p_verts[1:-1])
        self.d_verts = list(d_verts[1:-1])

        # translate fin such that top left is at 0,0,0 again
        xoff = self.p_verts[-1][-1][0]
        for vpair in self.p_verts:
            for v in vpair:
                v[0] -= xoff
        for vpair in self.d_verts:
            for v in vpair:
                v[0] -= xoff

        # Rays as quads between proximal and distal vertices
        for i, (p_pair, d_pair) in enumerate(
                zip(self.p_verts, reversed(self.d_verts))):
            # starting in proximal posterior corner anti-clockwise
            self.rays.append(
                (p_pair[0],
                 p_pair[1],
                 d_pair[0],
                 d_pair[1])
            )

        self.pflat = [i for pair in self.p_verts for i in pair]
        self.dflat = [i for pair in self.d_verts for i in pair]

    def grow(self):
        amount = 1.1  # ray length multiplier
        print(self.rays)
        # for ray in self.rays:
        #     ray[1][0] -= amount * np.cos(np.radians(self.theta))
        #     ray[1][1] -= amount * np.sin(np.radians(self.theta))
        for ray in self.rays:
            print(ray)
            ray[0][0] *= amount

            ray[1][0] *= amount
            ray[1][1] *= amount

    def plot_2d(self):
        """Plot the fin shape to check."""
        if len(self.p_verts) > self.nrays * 2:
            print("More vertices than expected. Possible 3d extrusion?")
            return

        plt.figure(figsize=(6, 3), layout="constrained")
        plt.title("Fin shape with rays")
        plt.plot([v[0] for v in self.pflat], [v[1] for v in self.pflat],
                 'o-', label='proximal')
        plt.plot([v[0] for v in self.dflat], [v[1] for v in self.dflat],
                 'o-', label='distal')
        print(self.rays)
        for ray in self.rays:
            ray = [(pt[0], pt[1]) for pt in ray]
            polygon = MplPolygon(ray, alpha=0.3)
            plt.gca().add_patch(polygon)
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.axis('equal')
        plt.legend()
        plt.grid()
        plt.show()

    def extrude_z(self, amount=0.1):
        """Extrude the fin shape in the z-direction."""

        verts = self.pflat + self.dflat

        # Extrude in the z direction
        self.verts_3d = [[x, y, z + amount/2] for (x, y, z) in verts] + \
            [[x, y, z - amount/2] for (x, y, z) in verts]

        # Create faces for the extruded mesh
        num_verts = len(verts)

        # Top and bottom faces
        self.faces.append([i for i in range(num_verts)])
        self.faces.append([i + num_verts for i in range(num_verts)])

        # Side faces
        for i in range(num_verts):
            next_i = (i + 1) % num_verts
            self.faces.append([i, next_i, next_i + num_verts, i + num_verts])

        vedo_mesh = Mesh([self.verts_3d, self.faces])
        vedo_mesh = vedo_mesh.triangulate()  # essential for yalla compatibility

        self.faces = vedo_mesh.cells
        self.verts_3d = vedo_mesh.vertices

        def find_vertex_index(arr, vertex):
            arr = np.asarray(arr)
            matches = np.where(
                np.all(np.isclose(arr, vertex, atol=1e-8), axis=1))[0]
            if len(matches) == 0:
                raise ValueError(f"Vertex {vertex} not found in verts_3d")
            return matches[0]

        # Ray faces
        for ray in self.rays:
            print("\n\n")
            print(ray)

            top = [[ray[0][0], ray[0][1], amount/2],
                   [ray[0][0], ray[0][1], -amount/2],
                   [ray[1][0], ray[1][1], -amount/2],
                   [ray[1][0], ray[1][1], amount/2]]

            bottom = [[ray[3][0], ray[3][1], amount/2],
                      [ray[3][0], ray[3][1], -amount/2],
                      [ray[2][0], ray[2][1], -amount/2],
                      [ray[2][0], ray[2][1], amount/2]]

            sides = [
                [top[0], top[1], bottom[1], bottom[0]],
                [top[1], top[2], bottom[2], bottom[1]],
                [top[2], top[3], bottom[3], bottom[2]],
                [top[3], top[0], bottom[0], bottom[3]],
            ]
            top_idx = [find_vertex_index(self.verts_3d, pt) for pt in top]
            bottom_idx = [find_vertex_index(
                self.verts_3d, pt) for pt in bottom]
            sides_idx = [
                [find_vertex_index(self.verts_3d, pt) for pt in quad] for quad in sides]
            self.ray_faces.append(top_idx)
            self.ray_faces.append(bottom_idx)
            self.ray_faces.extend(sides_idx)

        vedo_mesh = Mesh([self.verts_3d, self.ray_faces])
        vedo_mesh = vedo_mesh.triangulate()  # essential for yalla compatibility

        self.ray_faces = vedo_mesh.cells

    def plot_3d(self):
        """3D plot of the fin shape."""
        fin_mesh = Mesh([self.verts_3d, self.faces])
        ray_mesh = Mesh([self.verts_3d, self.ray_faces])  # .triangulate()
        p_pts = Points(self.pflat, r=12, c='blue')
        d_pts = Points(self.dflat, r=12, c='orange')

        plt = Plotter(axes=1)
        plt += fin_mesh.c('lightblue').alpha(0.5)  # .linecolor('black')
        plt += ray_mesh.c('orange').alpha(0.5)  # .linecolor('black')
        plt += p_pts
        plt += d_pts
        plt.show()

    def to_file(self, filename="init.vtk"):
        """Save vertices and rays to a text file."""

        # Create the vtk file
        with open(f"fin_{filename}", 'w', encoding='utf-8') as vtk_file:

            # Write the header
            vtk_file.write("# vtk DataFile Version 3.0\n")
            vtk_file.write("vtk output\n")
            vtk_file.write("ASCII\n")
            vtk_file.write("DATASET POLYDATA\n")

            # Write the vertices
            vtk_file.write("POINTS " + str(len(self.verts_3d)) + " float\n")
            for vert in self.verts_3d:
                vtk_file.write(str(vert[0]) + " " +
                               str(vert[1]) + " " + str(vert[2]) + "\n")

            # Write the faces
            vtk_file.write("POLYGONS " + str(len(self.faces)) +
                           " " + str(len(self.faces) + 4) + "\n")
            for face in self.faces:
                vtk_file.write(str(len(face)))
                for vert in face:
                    vtk_file.write(" " + str(vert))
                vtk_file.write("\n")
            # for ray in self.rays:
            #     vtk_file.write("LINE " + str(2) + " " +
            #                    str(self.verts_3d.index(ray[0])) + " " +
            #                    str(self.verts_3d.index(ray[1])) + "\n")
            # for ray in self.rays:
            #     for pt in ray:
            #         vtk_file.write()

        with open(f"ray_{filename}", 'w', encoding='utf-8') as ray_file:
            ray_file.write("# vtk DataFile Version 3.0\n")
            ray_file.write("vtk output\n")
            ray_file.write("ASCII\n")
            ray_file.write("DATASET POLYDATA\n")

            # Write the vertices
            ray_file.write("POINTS " + str(len(self.verts_3d)) + " float\n")
            for vert in self.verts_3d:
                ray_file.write(str(vert[0]) + " " +
                               str(vert[1]) + " " + str(vert[2]) + "\n")

            # Write the faces
            ray_file.write("POLYGONS " + str(len(self.ray_faces)) +
                           " " + str(len(self.ray_faces) * 5) + "\n")
            for face in self.ray_faces:
                ray_file.write(str(len(face)))
                for vert in face:
                    ray_file.write(" " + str(vert))
                ray_file.write("\n")


if __name__ == "__main__":
    fin = Fin()
    # fin.plot_2d()
    fin.extrude_z()
    fin.plot_3d()
    # fin.grow()
    fin.plot_2d()
    fin.to_file()
    # for i in range(10):
    #     fin.grow()
    #     fin.plot_2d()
    #     fin.extrude_z()
    #     fin.plot_3d_alt()

    # fin.plot_3d()
    # fin.to_file("fin_init.txt")
