from mayavi import mlab
import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.linalg import cg as conjugate_gradient
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csc_matrix
from scipy.linalg import block_diag

RADIUS_CONSTANT = 8

f_n = 20 #how many time steps until frame change = (time in s)/tau  ~0.6
tau = 1800/(f_n*10*1000) #time step increments
somite_chop_t = f_n*10000
T_n = f_n*5 #number of time steps = (time in s)/tau #10
HowManyUpdates = f_n #how many updates are printed (so you can see what's going on)
HowManyUpdates = min(HowManyUpdates, T_n)
HowManyPartUpdates = 5

saftey_barier = 100 #The grids for optimisation are made big enough to cover the whole PSM surface, plus saftey_barrier in all directions.

cell_df = pd.read_csv("Frames//17S_RF_PSM_Detailed.csv")
cell_df = cell_df[["ID","Position X Reference Frame","Position Y Reference Frame","Position Z Reference Frame"]]

testcells = -n #if you want to run a simulation with just the first m cells (eg for testing), set this to m
cells_px = cell_df["Position X Reference Frame"].to_numpy()[:testcells]
cells_py = cell_df["Position Y Reference Frame"].to_numpy()[:testcells]
cells_pz = cell_df["Position Z Reference Frame"].to_numpy()[:testcells]

N = np.shape(cells_px)[0] #number of non-border cells
PSM_cells = np.arange(N)

#Drasco Constants

radius = np.full(N,RADIUS_CONSTANT) #cell radii. Constant for now.
W = 10**(-5) #energy of adhesive contact (scalar)
v = np.full(N, 0.3) #?? guessed from supplements#poisson numbers (vector)
E_vector = np.full(N, 600) #young's moduli (Vector) #450 in paper
E = np.array([[1/((1-v[i]**2)/E_vector[i]+(1-v[j]**2)/E_vector[j]) for j in range(N)] for i in range(N)])
D = 0 #not given in paper #cell diffusion constant
gamma_ECM = 5*10**(4)#friction coefficent foa cell in the medium (for brownian motion)


gamma_perp_cc = 5*10**(4)
gamma_para_cc = 5*10**(3)
gamma_perp_bo = 0
gamma_para_bo = 0

f_bo = 2*10**(2)

#for optimisation
t_n = 1 #must be 1 or close to it

#Boundary Conditions
##avg_cells_n = 30 #number of avg cells
t_updateBC = 100#np.ceil(f_n/2)
cell_radius = 4.4
#border_force = 5*10**(-1)
#max_border_push = 5*10**(-2)

avg_h = 75 #cube side size for avarage cell grid

somite_stage = ["17_18","18_20"]
surface_boundary_frames = [int(c*10) for c in range(0,11)]
cwd = os.getcwd()

#for optimisation

def findPartition(x, y, z, grid_n, grid_borders, h): # Finds the partition (regarding the grid given) of the given cell. x,y,z = coords of cell
    new_cell = np.array([x,y,z]) - np.array([grid_borders[0][0],grid_borders[1][0],grid_borders[2][0]])
    [n_,m_,p_] = [int(np.floor(new_cell[0]/h)),
                  int(np.floor(new_cell[1]/h)), 
                  int(np.floor(new_cell[2]/h))]
    if n_>=grid_n[0] or m_>=grid_n[1] or p_>=grid_n[2] or n_<0 or m_<0 or p_<0:
        print("BIG ERROR in findPartition")
        print([n_,m_,p_])
        [n_,m_,p_] = grid_n-[1,1,1]
    return np.array([n_,m_,p_])

def updatePartition(grid_n, grid_borders, PSM_cells, h): #updates the partition variable. The partition variable holds the information for which cells are in each partition.
    partition = [[[[] for i in range(grid_n[2])] for i in range(grid_n[1])] for i in range(grid_n[0])]
    full_cubes = np.zeros((grid_n[0], grid_n[1], grid_n[2]), dtype=int)
    for cell_id in range(len(PSM_cells)):
        [n_,m_,p_] = findPartition(cells_px[PSM_cells[cell_id]],
                                   cells_py[PSM_cells[cell_id]],
                                   cells_pz[PSM_cells[cell_id]], grid_n, grid_borders, h)  
        if n_ < grid_n[0] and m_ < grid_n[1] and p_ < grid_n[2]: #ignore if cell is very far away; only happens if cell has escaped border
            partition[n_][m_][p_].append(cell_id)
            full_cubes[n_][m_][p_] = 1
        else:
            print("Problem in updatePartition (probably cells have gone too far away)")
    return [partition, full_cubes]

def setFrame(somite_stage, n, RADIUS_CONSTANT): #setFrame i.e update animation
    with open("Frames//"+somite_stage+"_"+ n + ".npy", 'rb') as f: #with open("Frames\\17_18_animation\\17_18_" + n + ".npy", 'rb') as f:
        PSM_x = np.load(f)
        PSM_y = np.load(f)
        PSM_z = np.load(f)
        triangles = np.load(f)
        neighbours = np.load(f)
        triangles = triangles.astype(int)
    for t in range(12):
        v1 = triangles[t][0]
        v2 = triangles[t][1]
        v3 = triangles[t][2]
        neighbours[v1,v2] = 1
        neighbours[v2,v1] = 1
        neighbours[v1,v3] = 1
        neighbours[v3,v1] = 1
        neighbours[v2,v3] = 1
        neighbours[v3,v2] = 1
    M = np.shape(PSM_x)[0]
    grid_borders = [[np.amin(PSM_x) - saftey_barier,
                     np.amax(PSM_x) + saftey_barier],
                    [np.amin(PSM_y) - saftey_barier,
                     np.amax(PSM_y) + saftey_barier],
                    [np.amin(PSM_z) - saftey_barier,
                     np.amax(PSM_z) + saftey_barier]]
    
    hash_grid_n = [int(np.ceil((grid_borders[0][1]-grid_borders[0][0])/RADIUS_CONSTANT)),int(np.ceil((grid_borders[1][1]-grid_borders[1][0])/RADIUS_CONSTANT)),int(np.ceil((grid_borders[2][1]-grid_borders[2][0])/RADIUS_CONSTANT))]
    hash_grid_avg_n = [int(np.ceil((grid_borders[0][1]-grid_borders[0][0])/avg_h)),
                       int(np.ceil((grid_borders[1][1]-grid_borders[1][0])/avg_h)),
                       int(np.ceil((grid_borders[2][1]-grid_borders[2][0])/avg_h))]
            
    return ([PSM_x, PSM_y, PSM_z, triangles, neighbours, grid_borders,
             hash_grid_n, hash_grid_avg_n, M])


#Boundary Conditions

def getInnerNormal(triangle, avg_cells, avg_full_cubes): #triangle as given from np.load(f), i.e entries in triangle should be integers
    #find normal
    v1 = np.array([PSM_x[triangle[0]], PSM_y[triangle[0]], PSM_z[triangle[0]]])
    v2 = np.array([PSM_x[triangle[1]], PSM_y[triangle[1]], PSM_z[triangle[1]]])
    v3 = np.array([PSM_x[triangle[2]], PSM_y[triangle[2]], PSM_z[triangle[2]]])
    n = np.cross(v1-v3,v2-v3)
    n_hat = n / np.linalg.norm(n)
    #####inner or outer normal?
    [n_,m_,p_] = findPartition(v1[0], v1[1], v1[2], hash_grid_avg_n, grid_borders, avg_h)
    
    surrounding_avg_cells_x = []
    surrounding_avg_cells_y = []
    surrounding_avg_cells_z = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                temp_x = min(max(0,n_-1+i),hash_grid_avg_n[0]-1)
                temp_y  = min(max(0,m_-1+j),hash_grid_avg_n[1]-1)
                temp_z  = min(max(0,p_-1+k),hash_grid_avg_n[2]-1)
                surrounding_avg_cells_x = np.union1d(surrounding_avg_cells_x, avg_cells[0][temp_x][temp_y][temp_z])
                surrounding_avg_cells_y = np.union1d(surrounding_avg_cells_y, avg_cells[1][temp_x][temp_y][temp_z])
                surrounding_avg_cells_z = np.union1d(surrounding_avg_cells_z, avg_cells[2][temp_x][temp_y][temp_z])
    
    test_point = [np.average(surrounding_avg_cells_x[np.where(surrounding_avg_cells_x!=0)]),
                  np.average(surrounding_avg_cells_y[np.where(surrounding_avg_cells_y!=0)]),
                  np.average(surrounding_avg_cells_z[np.where(surrounding_avg_cells_z!=0)])]
    #check if test_point is inside (should be if n is facing the right way)
    if np.dot(n,test_point-v1)>=0:
        return n_hat
    else:
        return -n_hat
    
def findAvgCells(partitionion, hash_grid_avg_n, full_cubes): #finds the average coordinates of all cells in given partition.
    avg_x = np.zeros(hash_grid_avg_n)
    avg_y = np.zeros(hash_grid_avg_n)
    avg_z = np.zeros(hash_grid_avg_n)
    non_empties = np.where(full_cubes)
    for n_ in range(hash_grid_avg_n[0]):
        for m_ in range(hash_grid_avg_n[1]):
            for p_ in range(hash_grid_avg_n[2]):
                if full_cubes[n_][m_][p_]:
                    avg_x[n_][m_][p_] = np.mean(cells_px[partitionion[n_][m_][p_]])
                    avg_y[n_][m_][p_] = np.mean(cells_py[partitionion[n_][m_][p_]])
                    avg_z[n_][m_][p_] = np.mean(cells_pz[partitionion[n_][m_][p_]])
                else:
                    [avg_x[n_][m_][p_], avg_y[n_][m_][p_], avg_z[n_][m_][p_]] = [0,0,0]
    return np.array([avg_x, avg_y, avg_z]) #radius

def greedyAlgo(x,y,z,part,greedy_vo): #https://en.wikipedia.org/wiki/Nearest_neighbor_search#Greedy_search_in_proximity_neighborhood_graphs
    v=int(greedy_vo[part[0]][part[1]][part[2]])
    escape=False
    while escape==False:
        v_distance = np.linalg.norm([PSM_x[v]-x, PSM_y[v]-y, PSM_z[v]-z])
        v_neighbours = np.where(neighbours[v]==1)[0]
        greedy_distances = np.array([np.linalg.norm([PSM_x[i]-x, PSM_y[i]-y, PSM_z[i]-z]) for i in v_neighbours])

        if min(greedy_distances) < v_distance:
            v = v_neighbours[np.argmin(greedy_distances)]
        else:
            escape=True
    return v

##### functions for Couzin cell-cell interactions (not used anymore but kept for legacy)
def findDHat(cell_i, test_area): #give r from cell i to all cell IDs. persepa_area = array of all cell ids in perception area (border cell ids come after cell ids)
    if cell_i>N:
        print("ERROR: findRHat i is greater than N")
        test_area = np.array(test_area)
    r_x = cells_px[test_area]- cells_px[cell_i]
    r_y = cells_py[test_area]- cells_py[cell_i]
    r_z = cells_pz[test_area]- cells_pz[cell_i]

    r_length = np.array([np.linalg.norm([r_x[i], r_y[i], r_z[i]]) for i in range(len(r_x))])

    [r_x,r_y,r_z]= np.where(r_length<=RADIUS_CONSTANT,[r_x,r_y,r_z],0)
    
    r_hat_x = np.divide(r_x, r_length, out=np.zeros(len(r_x)), where=r_length!=0)
    r_hat_y = np.divide(r_y, r_length, out=np.zeros(len(r_x)), where=r_length!=0)
    r_hat_z = np.divide(r_z, r_length, out=np.zeros(len(r_x)), where=r_length!=0)
    
    [d_x,d_y,d_z] = -np.sum([r_hat_x,r_hat_y, r_hat_z],axis=1)
    
    d_length = np.linalg.norm([d_x,d_y,d_z])
    [d_hat_x,d_hat_y,d_hat_z] = np.divide([d_x,d_y,d_z], d_length, out=np.zeros(3), where=d_length!=0)
    
    return np.array([d_hat_x,d_hat_y,d_hat_z])

def cellInteractionCouzin(cells_px,cells_py,cells_pz): #not used anymore
    dx = np.zeros(N)
    dy = np.zeros(N)
    dz = np.zeros(N)
    for n_ in range(hash_grid_n[0]):
        for m_ in range(hash_grid_n[1]):
            for p_ in range(hash_grid_n[2]):
                if main_non_empties[n_][m_][p_]:
                    #if t%int(T_n/HowManyUpdates)==0 and (((n_-1)*m_+(m_-1))*p_+p_)%(np.floor((hash_grid_n[0]*hash_grid_n[1]*hash_grid_n[2])/HowManyPartUpdates))==0:
                        #print("\t \t part: ", (((n_-1)*m_+(m_-1))*p_+p_)+1," of  ", hash_grid_n[0]*hash_grid_n[1]*hash_grid_n[2])
                    perception = []
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                temp_x = min(max(0,n_-1+i),hash_grid_n[0]-1)
                                temp_y  = min(max(0,m_-1+j),hash_grid_n[1]-1)
                                temp_z  = min(max(0,p_-1+k),hash_grid_n[2]-1)
                                perception = np.union1d(perception, partition[temp_x][temp_y][temp_z])
                    perception = [int(banana) for banana in np.unique(perception)]
                    for i in part:
                        [dx[i],dy[i],dz[i]] = findDHat(i, perception)
    cells_px += s*tau*dx
    cells_py += s*tau*dy
    cells_pz += s*tau*dz
    return(cells_px,cells_py, cells_pz)

####New Van Liedekerke/Drasdo Model
def drascoVelocities(r): #r[0] = cells_px etc #currently using constant radius (can implement pressue). plus border force
    r = np.array(r)
    r = np.ndarray.transpose(r)
    e = np.zeros((N,N,3))    #partition, main_non_empties
    e_product = np.zeros((N,N,3,3))
    G_ECM = np.zeros((N,3,3))
    G_cc  = np.zeros((N,N,3,3))
    G_bo  = np.zeros((N,3,3))
    d = np.array(euclidean_distances(r,r))
    R = np.array([[1/(1/radius[i] + 1/radius[j]) for j in range (N)] for i in range(N)])
    delta = [radius+radius[j] for j in range(N)]-d
    delta = np.where(delta>0,delta,0)
    for i in range(N):
        delta[i][i] = 0
        
    for n_ in range(hash_grid_n[0]):
        for m_ in range(hash_grid_n[1]):
            for p_ in range(hash_grid_n[2]):
                if main_non_empties[n_][m_][p_]:
                    perception = []
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                temp_x = min(max(0,n_-1+i),hash_grid_n[0]-1)
                                temp_y  = min(max(0,m_-1+j),hash_grid_n[1]-1)
                                temp_z  = min(max(0,p_-1+k),hash_grid_n[2]-1)
                                perception = np.union1d(perception, partition[temp_x][temp_y][temp_z])
                    perception = [int(banana) for banana in np.unique(perception)]
                    for i in perception:
                        for j in perception:
                            if np.linalg.norm(r[j]-r[i]) != 0: ##CHECK THIS
                                e[i][j] = (r[j]-r[i])/np.linalg.norm(r[j]-r[i])
                            else:
                                e[i][j] = [0, 0, 0]
                            e_product = np.outer(e[i][j],e[i][j])
                            G_cc[i][j] = gamma_perp_cc*e_product + gamma_para_cc*(np.identity(3) - e_product)
    e_border = np.zeros((N,3))
    G_bo = np.zeros((N,3))
    
    distances = np.zeros((N,1))
    for i in range(len(PSM_cells)):
        n = vertex_normals[closest_vertices[i]]
        closest_v = np.array([PSM_x[closest_vertices[i]], PSM_y[closest_vertices[i]], PSM_z[closest_vertices[i]]])
        distances[i] = np.linalg.norm(closest_v - r[i])
        dot = np.dot(n,np.array([cells_px[PSM_cells[i]], cells_py[PSM_cells[i]], cells_pz[PSM_cells[i]]]) - closest_v - n*cell_radius)
        if dot<=0: #part of cell is outside the border
            e_border[i] = (closest_v - r[i])/np.linalg.norm(closest_v - r[i])
        else:
            e_border[i]=[0,0,0]
        e_border_product = np.outer(e_border[i],e_border[i])
        G_bo = gamma_perp_bo*e_border_product + gamma_para_bo*(np.identity(3) - e_border_product)
    
    #Force
    F_rep = np.array(E*R**0.5*(delta**3)**0.5)
    F_adh = -np.pi*W*R
    e = np.swapaxes(e,0,2)
    F_cc = np.array([(F_rep+F_adh)*e[d] for d in range(3)])
    e = np.swapaxes(e,0,2)
    
    #F_mig
    A = 2*D*(gamma_ECM**2)#divided by 10 to make less effect
    F_mig = [np.random.multivariate_normal([0]*3,A*tau*np.identity(3)) for i in range(N)]
    F_mig = np.swapaxes(F_mig,0,1)
    F_bo_scalar = f_bo*distances**0.5 
    F_bo = np.swapaxes(F_bo_scalar*e_border,0,1) #GUESSED, CHANGE THIS
    
    
    F = np.sum(F_cc, axis=2) + F_mig + F_bo
    F = np.swapaxes(F,0,1)
    Force = F.flatten()
    
    #Friction
    G_ECM = [gamma_ECM*np.identity(3) for i in range(N)]
    #G_bo[i]=?
    A = [G_ECM[i] + np.sum(G_cc[i],axis=0) for i in range(N)] #G_bo
    Friction = np.zeros((N*3,N*3))
    for row in range(N):
        Friction[row*3:(row+1)*3] = np.block([
            np.array([-G_cc[row][column] for column in range(0,row)]).reshape((3, row*3)),
            A[row],
            np.array([-G_cc[row][column] for column in range(row+1,N)]).reshape((3, (N-row-1)*3))])
    
    start_conjugate_gradient = time.time()
    #preconditioner
    #P = np.zeros((N*3,N*3))
    #np.fill_diagonal(P, np.diag(Friction)) #Jacobi preconditioner
    Friction = csc_matrix(Friction)
    velocities = conjugate_gradient(Friction, Force)#, M = P) #last bit is for a preconditioner
    #print("cg took ", time.time() - start_conjugate_gradient)
    return (velocities[0][0::3], velocities[0][1::3], velocities[0][2::3])

somite_number=-1

dots_frame_x = np.zeros((T_n+1,N))
dots_frame_y = np.zeros((T_n+1,N))
dots_frame_z = np.zeros((T_n+1,N))
border_frame = np.zeros(T_n+1)

dots_frame_x[0] = cells_px[0:N]
dots_frame_y[0] = cells_py[0:N]
dots_frame_z[0] = cells_pz[0:N]
border_frame[0] = surface_boundary_frames[0]

start_time = time.time()
for t in np.arange(0,T_n):
    if t%int(T_n/HowManyUpdates)==0:
        print("Time step: ",t)
    if t%(f_n)==0: #Border update
        if t%somite_chop_t== 0:
            somite_number += 1
            frame_number = -1
        frame_number += 1
        print("\t Frame set to", somite_stage[somite_number],"....",surface_boundary_frames[frame_number], "( T =",t,")")


        [PSM_x, PSM_y, PSM_z, triangles, neighbours, grid_borders,
         hash_grid_n, hash_grid_avg_n,
         M] = setFrame(somite_stage[somite_number],str(surface_boundary_frames[frame_number]),RADIUS_CONSTANT)
        #print("\t part_n = ",part_n)

        #avg cells stuff hash_grid_n, grid_borders, PSM_cells
        [avg_cells_partition, avg_full_cubes] = updatePartition(hash_grid_avg_n, grid_borders, PSM_cells, avg_h)
        avg_cells = findAvgCells(avg_cells_partition, hash_grid_avg_n, avg_full_cubes)
        triangle_normals = np.array([getInnerNormal(t,avg_cells,avg_full_cubes) for t in triangles])
        vertex_normals = np.zeros((M,3))
        for i in range(M):
            vertex_normals[i] = np.average(triangle_normals[np.where(triangles==i)[0]],axis=0)
    greedy_vo = np.zeros((hash_grid_avg_n))
    if t%t_updateBC==0 or t%f_n==0: #update border condition stuff
        print("\t Updating BCs ( T =",t,")")
        BCs_begin = time.time()
        for i in range(hash_grid_avg_n[0]):
            for j in range(hash_grid_avg_n[1]):
                for k in range(hash_grid_avg_n[2]):
                    greedy_x = grid_borders[0][0] + avg_h*(i + 0.5)
                    greedy_y = grid_borders[1][0] + avg_h*(j + 0.5)
                    greedy_z = grid_borders[2][0] + avg_h*(k + 0.5)
                    greedy_vo[i][j][k] = int(np.argmin([np.linalg.norm([greedy_x-PSM_x[a],
                                                                        greedy_y-PSM_y[a],
                                                                        greedy_z-PSM_z[a]]) for a in range(M)]))
        closest_vertices = [greedyAlgo(cells_px[i], cells_py[i], cells_pz[i],
                                       findPartition(cells_px[i], cells_py[i], cells_pz[i],
                                                     hash_grid_avg_n, grid_borders, avg_h), greedy_vo) for i in PSM_cells]
        BCs_time = time.time() - BCs_begin
    if t%somite_chop_t== 0 and t!=0: #new somite has formed
        new_PSM_cells = np.where([isInPSM(i,closest_vertices,vertex_normals) for i in range(len(PSM_cells))],
                                 PSM_cells,
                                 np.ones(len(PSM_cells))*(-1))
        #redo closest_vertices
        closest_vertices = [closest_vertices[apple] for apple in np.where(new_PSM_cells != -1)[0]]
        PSM_cells = np.array([new_PSM_cells[apple] for apple in np.where(new_PSM_cells!= -1)[0]]).astype(int)

    if t%f_n == 0: #update optimisation partition stuff
        print("\t Updating Optimisation partitions ( T =",t,")")
        optimisation_begin = time.time()
        [partition, main_non_empties] = updatePartition(hash_grid_n, grid_borders, PSM_cells, RADIUS_CONSTANT)
        optimisation_time = time.time() - optimisation_begin


    if t%int(T_n/HowManyUpdates)==0:
        print("\t Calculating cell movements")
    cell_movements_begin = time.time()


    #cell-cell interaction
    velocities = drascoVelocities([cells_px,cells_py,cells_pz])

    #print(np.swapaxes(np.array([cells_px, cells_py, cells_pz]),0,1))
    cells_px += velocities[0]
    cells_py += velocities[1]
    cells_pz += velocities[2]
    ##cell-border interaction (old version. Still working if you uncomment)
    #for i in range(len(PSM_cells)):
    #    n = vertex_normals[closest_vertices[i]]
    #    closest_v = np.array([PSM_x[closest_vertices[i]], PSM_y[closest_vertices[i]], PSM_z[closest_vertices[i]]])
    #    dot = np.dot(n,np.array([cells_px[PSM_cells[i]], cells_py[PSM_cells[i]], cells_pz[PSM_cells[i]]]) - closest_v)
    #    if dot<0:
    #        #cell has gone outside the border
    #        [cells_px[PSM_cells[i]],
    #         cells_py[PSM_cells[i]],
    #         cells_pz[PSM_cells[i]]] = [cells_px[PSM_cells[i]],
    #                                    cells_py[PSM_cells[i]],
    #                                    cells_pz[PSM_cells[i]]] + min(np.abs(dot*border_force),
    #                                                                   max_border_push)*tau*n

    dots_frame_x[t+1] = cells_px[0:N]
    dots_frame_y[t+1] = cells_py[0:N]
    dots_frame_z[t+1] = cells_pz[0:N]

    #update partition after movements
    [partition, main_non_empties] = updatePartition(hash_grid_n, grid_borders, PSM_cells, RADIUS_CONSTANT)

    border_frame[t+1] = surface_boundary_frames[frame_number]
    cell_movements_time = time.time() - cell_movements_begin
    if t%np.ceil(T_n/HowManyUpdates)==0:
        if t==0:
            now = time.strftime("%H:%M:%S", time.localtime())
            timeuntilend = int((cell_movements_time + BCs_time*(1/t_updateBC) + optimisation_time*(1/t_n))*T_n/60)
            print("--- Current time: ", now, "---")
            print("--- Estimated time until end: ",timeuntilend,"minutes ---")


print("---", int((time.time() - start_time))/60,"minutes ---")

tracks_x = pd.DataFrame(data=dots_frame_x,index=range(T_n+1),columns=range(N)) 
tracks_y = pd.DataFrame(data=dots_frame_y,index=range(T_n+1),columns=range(N)) 
tracks_z = pd.DataFrame(data=dots_frame_z,index=range(T_n+1),columns=range(N)) 
tracks_frame = pd.DataFrame(data=border_frame)

tracks_x.to_csv(cwd+'/saves/tracks_x'+'_'+str(0)+'_'+str(0)+'.csv')
tracks_y.to_csv(cwd+'/saves/tracks_y'+'_'+str(0)+'_'+str(0)+'.csv')
tracks_z.to_csv(cwd+'/saves/tracks_z'+'_'+str(0)+'_'+str(0)+'.csv')
tracks_frame.to_csv(cwd+'/saves/tracks_frame'+'_'+str(0)+'_'+str(0)+'.csv')

print("done")

from shutil import make_archive
make_archive(cwd+'/saves',
                    'zip',
                    cwd,
                    'saves')
