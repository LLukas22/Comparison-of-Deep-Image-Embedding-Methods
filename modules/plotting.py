from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from openTSNE import TSNE
from tqdm import tqdm
import os

def get_cmap(n, name='jet'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_tsne(embeddings:np.ndarray,labels:np.ndarray)->np.ndarray:
    classes = max(np.unique(labels))+1
    new_map = get_cmap(classes)
    tsne = TSNE(n_jobs=os.cpu_count())
    tsne_proj = tsne.fit(embeddings)
    for lab in range(classes):
        indices = labels==lab
        points = tsne_proj[indices]
        color = new_map(lab/float(classes))
        plt.scatter(points[:,0],points[:,1],color=color,alpha = 0.8, s=3.5)
    return tsne_proj
    
    
def plot_results(results:dict[str,tuple[np.ndarray,np.ndarray]],name:str,size:int=10)->None:
    plt.figure(figsize=(size*len(results),size))
    for i,key in tqdm(enumerate(results),"Building T-SNE plots"):
        plt.subplot(1,len(results),i+1)
        embeddings,labels = results[key]
        plot_tsne(embeddings,labels)
        plt.axis("off")
        plt.title(key, fontsize=22)
        
    plt.tight_layout()
    plt.savefig(f'plots/{name}.png', bbox_inches='tight')

