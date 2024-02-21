%matplotlib notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

t_n = 10 #update every t_n timesteps
l = 300 #min length of each partition ( actual partition length will be the smallest 
               #length larger than l that divides the partition line length)
          
run = '1_0'

tracks_x = pd.read_csv('tracks_x_'+run+'.csv').to_numpy()[:,1:]
tracks_y = pd.read_csv('tracks_y_'+run+'.csv').to_numpy()[:,1:]
tracks_z = pd.read_csv('tracks_z_'+run+'.csv').to_numpy()[:,1:]

[T_n, N] = np.shape(tracks_x)

def setFrame(n):
    with open("Frames\\17_18_animation\\17_18_" + n + ".npy", 'rb') as f:
        PSM_x = np.load(f)
        PSM_y = np.load(f)
        PSM_z = np.load(f)
        triangles = np.load(f)
        neighbours = np.load(f)
        triangles = triangles.astype(int)
    M = np.shape(PSM_x)[0]

    distances = np.array([np.linalg.norm([PSM_x[i], PSM_y[i], PSM_z[i]]) for i in range(M)])
    furthest_border_cell_id = np.where(distances==np.max(np.abs(distances)))[0][0]
    A_end = np.array([PSM_x[furthest_border_cell_id],PSM_y[furthest_border_cell_id],PSM_z[furthest_border_cell_id]])
    
    distances_from_ant = np.array([np.linalg.norm([PSM_x[i], PSM_y[i], PSM_z[i]]-A_end) for i in range(M)])
    furthest_border_cell_id_posterior = np.where(distances_from_ant==np.max(np.abs(distances_from_ant)))[0][0]
    P_end = np.array([PSM_x[furthest_border_cell_id_posterior],PSM_y[furthest_border_cell_id_posterior],PSM_z[furthest_border_cell_id_posterior]])
    partition_line = A_end - P_end
    
    part_line_length = np.linalg.norm(partition_line)
    part_n = max(int(np.floor(part_line_length/l)),1) #number of partitions

    costheta = partition_line[0]/np.linalg.norm(partition_line)
    sintheta = -partition_line[1]/np.linalg.norm(partition_line)
    rot_matrix = np.matrix([[costheta, -sintheta, 0], [sintheta, costheta, 0],[0, 0, 1]])

    return ([PSM_x, PSM_y, PSM_z, triangles, neighbours, rot_matrix, part_line_length, M, P_end])

[PSM_x, PSM_y, PSM_z, triangles, 
 neighbours, rot_matrix, 
 part_line_length, M, P_end] = setFrame('0')

start_time=0
wait_time=88
T_n = wait_time - start_time
k=10;
x_n = int(N/2)
test_cells = np.random.randint(N, size=x_n)

X = [[[tracks_x[t][x],tracks_y[t][x],tracks_z[t][x]] for x in test_cells] for t in range(T_n)]
Y = [[[tracks_x[t][n],tracks_y[t][n],tracks_z[t][n]] for n in range(N)] for t in range(T_n)]
distance = [euclidean_distances(X[t], Y[t]) for t in range(T_n)]

change_in_neighbours = np.zeros(x_n)
                
neighbourhood = [[np.argpartition(distance[t][x], k+1)[:k+1] for t in range(T_n)] for x in range(x_n)]
change_in_neighbours = [np.shape(np.setdiff1d(np.unique(neighbourhood[x]),neighbourhood[x][0]))[0] 
                        for x in range(x_n)]
                        
start_x_AP = [(np.matmul(rot_matrix, [tracks_x[start_time][test_cells[i]],
                                      tracks_y[start_time][test_cells[i]],
                                      tracks_z[start_time][test_cells[i]]]) - P_end)[0,0] for i in range(x_n)]

#[36,48]
#sa = [24,30]
plt.plot(start_x_AP,change_in_neighbours, 'o', markersize=1)
plt.ylabel('Change in Neighbours')
plt.xlabel('Start position (A-P Axis)')
plt.title('Nearest neighbours for $s_p = 88$, $s_a = 30$, k=' + str(k))
plt.show()
