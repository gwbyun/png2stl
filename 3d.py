#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:31:11 2022

@author: gw
"""

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.data import horse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import version
import plotly as pl
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import figure_factory as FF
from plotly.graph_objs import *
from stl import mesh


def make_mesh(image, threshold=255, step_size=1):

    print ("Transposing surface")
    p = image.transpose(2,1,0)
    
    print ("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print ("Drawing")
    
    # Make the colormap single color since the axes are positional not intensity. 
#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    plot(fig)

def plt_3d(verts, faces):
    print ("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.show()

#v, f = make_mesh(imgs_after_resamp, 350) 
#plt_3d(v, f)



def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir('edge_data')                                                 


if __name__=="__main__":
    """Dataset path"""
    dataset_path = "test"
    dataset = sorted(glob(os.path.join(dataset_path,"*.png")))
    imgs = np.empty((len(dataset), 512, 512), dtype=np.float32)
    i=0
    for data in dataset:
        print(data)
       
        imgs[i] = cv2.imread(data,cv2.IMREAD_GRAYSCALE)
        i = i+1bgw
    print(np.max(imgs))    
    print(np.nonzero(imgs))
     
    v, f = make_mesh(imgs, 255.0) 
    #plt_3d(v, f)
    plotly_3d(v, f)
    
    cube = mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(f):
        for j in range(3):
            cube.vectors[i][j] = v[f[j],:]

    # Write the mesh to file "cube.stl"
    cube.save('CT.stl')















