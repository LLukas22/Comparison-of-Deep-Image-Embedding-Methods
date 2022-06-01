import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class KNN():
    def __init__(self,model:torch.nn.Module,dataloader:DataLoader,save_path:str) -> None:
        
        
        self.labels = None
        self.embeddings = None
        knn_dir = os.path.join(save_path,'knn')
        if not os.path.exists(knn_dir):
            os.makedirs(knn_dir)
            
        if os.path.isfile(os.path.join(knn_dir,"embeddings.pk")):
            self.embeddings = pickle.load(open(os.path.join(knn_dir,"embeddings.pk"),"rb"))
        
        if os.path.isfile(os.path.join(knn_dir,"labels.pk")):
            self.labels = pickle.load(open(os.path.join(knn_dir,"labels.pk"),"rb"))
        
        if self.labels is None or self.embeddings is None:
            model = model.to(DEVICE)
            model.eval()
            labels = []
            embeddings = []
            with torch.no_grad():
                for batch in tqdm(dataloader,"Building Embeddings"):
                    img,label = batch
                    embedding = model(img.to(DEVICE)).cpu().numpy()
                    embeddings.append(embedding)
                    labels.append(label)
                    
            self.labels = np.vstack(labels)
            self.embeddings = np.vstack(embeddings).astype('float32')
            pickle.dump(self.embeddings,open(os.path.join(knn_dir,"embeddings.pk"),"wb"))
            pickle.dump(self.labels,open(os.path.join(knn_dir,"labels.pk"),"wb"))
            
        self.dimensionality = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimensionality) 
        self.index.add(self.embeddings)
        
    def querry(self,embedding:np.ndarray,k:int=5) -> np.ndarray:
        """
        Querry the embedding index for the k nearest neighbors of the embedding
        """
        return self.index.search(embedding,k)[1]
            
    def querry_labels(self,embedding:np.ndarray,k:int=5) -> np.ndarray:
        assert k > 1, "K hast to be greater than 1!"
        indices = self.querry(embedding,k=k)
        predictions = np.squeeze(self.labels[indices])
        return self._bincount_2D(predictions)    
            
    def _bincount_2D(self,array:np.ndarray):
        N,M = array.shape[0], np.max(array)+1
        helper = np.zeros(shape=(N, M), dtype=int)
        advanced_indexing = np.repeat(np.arange(N), array.shape[1]), array.ravel()
        np.add.at(helper, advanced_indexing, 1)
        return (M-1)-np.argmax(helper[:,::-1], axis=1)