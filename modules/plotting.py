from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from openTSNE import TSNE

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_tsne(embeddings:np.ndarray,labels:np.ndarray,classes:int,max_points_per_class:int=20):
    new_map = get_cmap(classes)
    tsne = TSNE(n_jobs=4)
    tsne_proj = tsne.fit(embeddings)
    for lab in range(classes):
        indices = labels==lab
        points = tsne_proj[indices]
        if len(points) > max_points_per_class:
            points = points[:max_points_per_class]
        color = new_map(lab/float(classes))
        plt.scatter(points[:,0],points[:,1],color=color)
    

