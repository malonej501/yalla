from mayavi import mlab
import numpy as np
import pandas as pd

run = '0_0'
somite_stage = ["17_18"]#["17_18", "18_20"]

cell_size_animation = 10

def setFrame(somite_stage,n):
    with open("Frames\\"+somite_stage+"_animation\\"+somite_stage+"_"+ n + ".npy", 'rb') as f:
        PSM_x = np.load(f)
        PSM_y = np.load(f)
        PSM_z = np.load(f)
        triangles = np.load(f)
        neighbours = np.load(f)
        triangles = triangles.astype(int)
    return ([PSM_x, PSM_y, PSM_z, triangles])

    
tracks_x = pd.read_csv('tracks_x_'+run+'.csv')
tracks_y = pd.read_csv('tracks_y_'+run+'.csv')
tracks_z = pd.read_csv('tracks_z_'+run+'.csv')
tracks_frame = pd.read_csv('tracks_frame_'+run+'.csv')
#tracks_PSM_cells = pd.read_csv('PSM_cells_frame_'+run+'.csv')

dots_frame_x = tracks_x.to_numpy()
dots_frame_y = tracks_y.to_numpy()
dots_frame_z = tracks_z.to_numpy()
border_frame = tracks_frame.to_numpy()
#PSM_cells = tracks_PSM_cells.to_numpy()


dots_frame_x = dots_frame_x[:,1:]
dots_frame_y = dots_frame_y[:,1:]
dots_frame_z = dots_frame_z[:,1:]
border_frame = border_frame[:,1]/10
border_frame = border_frame.astype(int)
#PSM_cells = PSM_cells[:,1:]
#PSM_cells = PSM_cells.astype(int)


T_n = np.shape(dots_frame_x)[0]
N = max([np.shape(dots_frame_x[i])[0] for i in range(len(border_frame))])
PSM_cells = [np.arange(N) for i in range(T_n)]



#border_frame = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10 ,10, 10 ,11, 11 ,11,
#                12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19]

M = max([np.shape(setFrame(somite_stage[s],str(0))[0])[0] for s in range(len(somite_stage))])

red_dot_1 = np.intersect1d(np.intersect1d(np.where(dots_frame_x[1,:] > 50),np.where(dots_frame_x[1,:] < 70)),
                          np.intersect1d(np.where(dots_frame_y[1,:] > -10),np.where(dots_frame_y[1,:] < 10)))

red_dot_2 = np.intersect1d(np.intersect1d(np.where(dots_frame_x[1,:] > 130),np.where(dots_frame_x[1,:] < 150)),
                          np.intersect1d(np.where(dots_frame_y[1,:] > 0),np.where(dots_frame_y[1,:] < 20)))

red_dot_3 = np.intersect1d(np.intersect1d(np.where(dots_frame_x[1,:] > 230),np.where(dots_frame_x[1,:] < 250)),
                          np.intersect1d(np.where(dots_frame_y[1,:] > -70),np.where(dots_frame_y[1,:] < -50)))

redcells = np.union1d(np.union1d(red_dot_1, red_dot_2), red_dot_3)

#redcells = np.array(0)
whitecells = np.setdiff1d(np.array(range(N)), redcells, assume_unique=True)

[T_n, N] = np.shape(dots_frame_x)
PSM_x = [setFrame(somite_stage[0],str(n))[0] for n in range(0,110,10)]
PSM_y = [setFrame(somite_stage[0],str(n))[1] for n in range(0,110,10)]
PSM_z = [setFrame(somite_stage[0],str(n))[2] for n in range(0,110,10)]
triangles = [setFrame(somite_stage[0],str(n))[3] for n in range(0,110,10)]

@mlab.animate(delay = 100) 
def updateAnimation(): 
    while True:
        t = 0
        for t in range(T_n):
            whitedots.mlab_source.set(x = dots_frame_x[t][np.intersect1d(whitecells,PSM_cells[t])],
                                      y = dots_frame_y[t][np.intersect1d(whitecells,PSM_cells[t])],
                                      z = dots_frame_z[t][np.intersect1d(whitecells,PSM_cells[t])])
            reddots.mlab_source.set(x = dots_frame_x[t][np.intersect1d(redcells,PSM_cells[t])],
                                    y = dots_frame_y[t][np.intersect1d(redcells,PSM_cells[t])],
                                    z = dots_frame_z[t][np.intersect1d(redcells,PSM_cells[t])])
            border.mlab_source.set(x = PSM_x[border_frame[t]], y=PSM_y[border_frame[t]], z=PSM_z[border_frame[t]], triangles = triangles[border_frame[t]])
            yield
mlab.figure(bgcolor = (0.3,0.3,0.3))
border = mlab.triangular_mesh(PSM_x[0], PSM_y[0], PSM_z[0], triangles[0], color= (0,0.5,1), opacity=0.2)
whitedots = mlab.points3d(dots_frame_x[0][np.intersect1d(whitecells,PSM_cells[0])],
                          dots_frame_y[0][np.intersect1d(whitecells,PSM_cells[0])],
                          dots_frame_z[0][np.intersect1d(whitecells,PSM_cells[0])], color = (1,1,1), scale_factor=cell_size_animation)
reddots = mlab.points3d(dots_frame_x[0][np.intersect1d(redcells,PSM_cells[0])],
                        dots_frame_y[0][np.intersect1d(redcells,PSM_cells[0])],
                        dots_frame_z[0][np.intersect1d(redcells,PSM_cells[0])], color = (1,0,0), scale_factor=cell_size_animation)


updateAnimation() 
mlab.show()

