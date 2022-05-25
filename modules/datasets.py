import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import os
import random
from PIL import Image
from torchvision import transforms
from pathlib import Path
import torchdatasets as td
from tqdm import tqdm


def _fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(_fast_scandir(dirname))
    return subfolders
        
def _get_image_directories(root_dir):
    """
    Returns a list of directories with images in the root directory
    """
    return [dir for dir in _fast_scandir(root_dir) if os.walk(dir).__next__()[2]]

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    
class IndexRange(object):
    def __init__(self,start:int,stop:int) -> None:
        self.start = start
        self.stop = stop
        
    def inRange(self,index:int) -> bool:
        return index >= self.start and index < self.stop
         
     
class CachingDataset(Dataset):
    def __init__(self,dataset:Dataset,cache_in_ram:bool = False,cache_path:str = None) -> None:
        self.dataset = dataset
        if cache_path:
            self.dataset = td.datasets.WrapDataset(self.dataset).cache(td.cachers.Pickle(Path(cache_path)))
        if cache_in_ram:
            self.dataset = td.datasets.WrapDataset(self.dataset).cache()
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
        
        
class ImageDataset(Dataset):
    def __init__(self,
                 dir:str,
                 transform:torch.nn.Sequential=transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                 label:int = None,
                 random_seed:int = 42,
                 max_size:int=None) -> None:

            self.random_provider = random.Random(random_seed)
            self.dir = dir
            self.transform = transform
            self.label = label
            self.files = []
            self.all_files = os.listdir(self.dir)
            self.all_files.sort()
            
            if max_size and max_size > 0:
                if len(self.all_files) > max_size:
                    self.all_files = self.random_provider.sample(self.all_files,k=max_size)
                    
            for file in self.all_files:
                self.files.append(os.path.join(self.dir, file))
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path = self.files[idx]
        #ensure the images are RGB
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.label is not None:
            return image, torch.tensor([self.label]) 
        else:
            return image
        
class MultiLabelDataset(Dataset):
    def __init__(self,
                 dir:str,
                 transform:torch.nn.Sequential=transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                 cache_path:str = None,
                 cache_in_ram:bool=False,
                 random_seed:int = 42,
                 max_size_per_class:int = None) -> None:
        
        self.dir = dir
        self.labels = {}
        self.files = []
        self.datasets={}
        self.ranges:dict[IndexRange,str]={}
        
        directories = _get_image_directories(self.dir)
        for directory in directories:
            label = os.path.basename(directory)
            start_index = len(self.files)
            self.labels[label] = len(self.labels)
            dataset = ImageDataset(directory,transform,label=self.labels[label],random_seed=random_seed,max_size=max_size_per_class)
            self.files += dataset.files
            
            dataset = CachingDataset(dataset,cache_in_ram,os.path.join(cache_path,label) if cache_path else None)
            self.datasets[label] = dataset
            self.ranges[IndexRange(start_index,len(self.files))] = label            

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        for range in self.ranges:
            if range.inRange(idx):
                return self.datasets[self.ranges[range]][idx - range.start]
            
def get_random(lst,random_provider:random.Random):
    idx = random_provider.randrange(0, len(lst))
    return lst.pop(idx)

class ContrastiveDataset(Dataset):
    def __init__(self,multiLabelDataset:MultiLabelDataset,random_seed:int = 42) -> None:
        
        self.multiLabelDataset = multiLabelDataset
        self.random_provider = random.Random(random_seed)
        self.possible_indices = list(range(len(multiLabelDataset)))
        while len(self.possible_indices)%2 == 1:
            self.possible_indices.remove(self.possible_indices[self.random_provider.randrange(0, len(self.possible_indices))])

            
        self.image_pairs = []
        
        for index in self.possible_indices:
            other_index = get_random(self.possible_indices,self.random_provider)
            while other_index == index:
                other_index = get_random(self.possible_indices,self.random_provider)
            self.image_pairs.append((index,other_index))
            
    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        index1,index2 = self.image_pairs[idx]
        image1,label1 =  self.multiLabelDataset[index1]
        image2,label2 =  self.multiLabelDataset[index2]
        
        return image1,image2,torch.eq(label1, label2).float()
            
              
                
if __name__ == "__main__":
    dataset = MultiLabelDataset(os.path.abspath("./Carparts"),max_size_per_class=100)
    contrastiveDataset = ContrastiveDataset(dataset)
    print(len(contrastiveDataset))