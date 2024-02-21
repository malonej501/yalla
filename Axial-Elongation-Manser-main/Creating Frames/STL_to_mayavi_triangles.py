%gui qt
from mayavi import mlab
import numpy as np
from stl import mesh
filename = "18_20_80"

print(filename)

my_mesh = mesh.Mesh.from_file(filename + ".stl") #17_to_18_f1 #Cube_3d #Stanford_Bunny_sample #PSM17S_aligned
points = np.unique(my_mesh.vectors.reshape([int(my_mesh.vectors.size/3), 3]), axis=0)
#print(points,"\n")
n_points=points.shape[0]
n_triangles=my_mesh.vectors.shape[0]

x = [points[i,0] for i in range(0,n_points)]
y = [points[i,1] for i in range(0,n_points)]
z = [points[i,2] for i in range(0,n_points)]

print("n_points: ",n_points,"\nn_triangles:",n_triangles)

new_triangles = np.ones((n_triangles,3))*-1

point_i=0
for point in points:
    #if point_i%1000 == 0:
    #    print("point number: ",point_i)
    indicator = np.all((my_mesh.vectors-np.array(point))==0, axis=2)
    new_triangles = np.where(indicator==1,point_i,new_triangles)
    point_i=point_i+1
new_triangles = new_triangles.astype(int)

#make neighbour matrix

neighbours = np.zeros((n_points,n_points))
for t in range(n_triangles):
    v1 = new_triangles[t][0]
    v2 = new_triangles[t][1]
    v3 = new_triangles[t][2]
    neighbours[v1,v2] = 1
    neighbours[v2,v1] = 1
    neighbours[v1,v3] = 1
    neighbours[v3,v1] = 1
    neighbours[v2,v3] = 1
    neighbours[v3,v2] = 1
    
    new_triangles = new_triangles.astype(int)
cube=mlab.triangular_mesh(x, y, z, new_triangles)
#mlab.show()
with open(filename + ".npy", 'wb') as f:
    np.save(f, x)
    np.save(f,y)
    np.save(f,z)
    np.save(f,new_triangles)
    np.save(f,neighbours)
    
print("done")
print(np.shape(neighbours))
