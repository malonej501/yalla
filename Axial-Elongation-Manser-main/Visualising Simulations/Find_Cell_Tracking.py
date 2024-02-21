from mayavi import mlab
import numpy as np
import pandas as pd

mlab.options.offscreen = True
mlab.figure(bgcolor = (0.6,0.6,0.6))
f = mlab.gcf()
camera = f.scene.camera
camera.pitch(120)
camera.yaw(20)

def setFrame(n):
    with open("Frames\\17_18_animation\\17_18_" + n + ".npy", 'rb') as f:
        PSM_x = np.load(f)
        PSM_y = np.load(f)
        PSM_z = np.load(f)
        triangles = np.load(f)
        neighbours = np.load(f)
        triangles = triangles.astype(int)
    return ([PSM_x, PSM_y, PSM_z, triangles])

tracks_x = pd.read_csv('tracks_x_0_0.csv')
tracks_y = pd.read_csv('tracks_y_0_0.csv')
tracks_z = pd.read_csv('tracks_z_0_0.csv')

dots_frame_x = tracks_x.to_numpy()
dots_frame_y = tracks_y.to_numpy()
dots_frame_z = tracks_z.to_numpy()
dots_frame_x = dots_frame_x[:,1:]
dots_frame_y = dots_frame_y[:,1:]
dots_frame_z = dots_frame_z[:,1:]

x0 = dots_frame_x[0]
y0 = dots_frame_y[0]
z0 = dots_frame_z[0]

[T_n, N] = np.shape(dots_frame_x)
N=N-1
T_n=T_n-1
f_n = int(T_n/11)

red_dot_1 = np.intersect1d(np.intersect1d(np.intersect1d(np.where(x0 > 50),np.where(x0 < 70)),
              np.intersect1d(np.where(y0 > -10),np.where(y0 < 10))),
              np.intersect1d(np.where(z0 > -100), np.where(z0<-20)))

red_dot_2 = np.intersect1d(np.intersect1d(np.intersect1d(np.where(x0 > 130),np.where(x0 < 150)),
              np.intersect1d(np.where(y0 > 0),np.where(y0 < 20))),
               np.intersect1d(np.where(z0 > -100), np.where(z0<-30)))

red_dot_3 = np.intersect1d(np.intersect1d(np.intersect1d(np.where(x0 > 230),np.where(x0 < 250)),
              np.intersect1d(np.where(y0 > -70),np.where(y0 < -50))),
               np.intersect1d(np.where(z0 > -100), np.where(z0<-30)))

redcells = np.union1d(np.union1d(red_dot_1, red_dot_2), red_dot_3)
whitecells = np.setdiff1d(np.array(range(N)), redcells, assume_unique=True)

for frame in [2,4,6,8,10]:#range(0,11,2):
    print(frame)
    [PSM_x, PSM_y, PSM_z, triangles] = setFrame(str(frame*10))
    for i in [0,1]:
        for j in [0,1]:
            run = str(i)+'_'+str(j)
            print(run)
        
            tracks_x = pd.read_csv('tracks_x_'+run+'.csv')
            tracks_y = pd.read_csv('tracks_y_'+run+'.csv')
            tracks_z = pd.read_csv('tracks_z_'+run+'.csv')

            dots_frame_x = tracks_x.to_numpy()
            dots_frame_y = tracks_y.to_numpy()
            dots_frame_z = tracks_z.to_numpy()
            dots_frame_x = dots_frame_x[:,1:][frame*(f_n+1)-2]
            dots_frame_y = dots_frame_y[:,1:][frame*(f_n+1)-2]
            dots_frame_z = dots_frame_z[:,1:][frame*(f_n+1)-2]

            border = mlab.triangular_mesh(PSM_x, PSM_y, PSM_z, triangles, color= (0,0.5,1), opacity=0.2)
            whitedots = mlab.points3d(dots_frame_x[whitecells], dots_frame_y[whitecells], dots_frame_z[whitecells], color = (1,1,1), scale_factor=3)
            reddots = mlab.points3d(dots_frame_x[redcells], dots_frame_y[redcells], dots_frame_z[redcells], color = (1,0,0), scale_factor=3)
            filename = 'red_tracks'+run+'_frame_'+str(frame)+'.png'
            mlab.savefig(size=(2000, 2000), filename=filename, magnification='auto')
            mlab.clf()
