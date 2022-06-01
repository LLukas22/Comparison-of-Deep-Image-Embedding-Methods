from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_tsne(embeddings:np.ndarray,labels:np.ndarray,classes:int):
    new_map = get_cmap(classes)
    tsne = TSNE(2)
    tsne_proj = tsne.fit_transform(embeddings)
    for lab in range(classes):
        indices = labels==lab
        color = new_map(lab/float(classes))
        plt.scatter(tsne_proj[indices,0],tsne_proj[indices,1],color=color)
    

