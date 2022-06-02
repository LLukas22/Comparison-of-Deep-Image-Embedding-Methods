import torch
from torch.utils.data import Dataset
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
    possible_dirs =  [dir for dir in _fast_scandir(root_dir) if os.walk(dir).__next__()[2]]
    return [possible_dir for  possible_dir in possible_dirs if len(_fast_scandir(possible_dir))==0]

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
         

class AugmentingDataset(Dataset):
    def __init__(self,sourceDataset:Dataset,augmentation:torch.nn.Sequential,transforms:torch.nn.Sequential,factor:int=4) -> None:
        super(AugmentingDataset,self).__init__()
        self.sourceDataset = sourceDataset
        self.factor = factor
        self.transforms = transforms
        self.augmentation = augmentation
        
    def __len__(self) -> int:
        return len(self.sourceDataset)*self.factor
    
    def __getitem__(self, idx):
        if idx >= len(self):
            return
        
        if idx/len(self.sourceDataset) < 1:
            img,label = self.sourceDataset[idx]
            return self.transforms(img),label
        else:
            index = idx%len(self.sourceDataset)
            img,label = self.sourceDataset[index]
            return self.transforms(self.augmentation(img)),label

        
         
class CachingDataset(Dataset):
    def __init__(self,dataset:Dataset,cache_in_ram:bool = False,cache_path:str = None) -> None:
        super(CachingDataset,self).__init__()
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
            super(ImageDataset,self).__init__()
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
                 max_size_per_class:int = None,
                 max_classes:int=None,
                 label_index:int=0) -> None:
        super(MultiLabelDataset,self).__init__()
        self.dir = dir
        self.labels = {}
        self.files = []
        self.datasets={}
        self.ranges:dict[IndexRange,str]={}
        self.reversed_ranges:dict[str,IndexRange]={}
        self.random_provider = random.Random(random_seed)
        
        found_directories = _get_image_directories(self.dir)
        directories = []
        #reduce the classes if needed
        if max_classes and len(found_directories)>max_classes:
            for i in range(max_classes):
                directories.append(found_directories.pop(self.random_provider.randrange(0, len(found_directories))))
        else:
            directories = found_directories        
               
        for directory in directories:
            relative_path = os.path.relpath(directory,self.dir)
            label = os.path.split(relative_path)[label_index]
            start_index = len(self.files)
            self.labels[label] = len(self.labels)
            dataset = ImageDataset(directory,transform,label=self.labels[label],random_seed=random_seed,max_size=max_size_per_class)
            if len(dataset) == 0:
                continue
            self.files += dataset.files
            
            dataset = CachingDataset(dataset,cache_in_ram,os.path.join(cache_path,label) if cache_path else None)
            self.datasets[label] = dataset
            self.ranges[IndexRange(start_index,len(self.files))] = label   
        
        self.reversed_ranges =  dict((reversed(item) for item in self.ranges.items()))  

    def __len__(self):
        return len(self.files)

    def getSubDataset(self,idx:int)->str:
         for range in self.ranges:
            if range.inRange(idx):
                return self.ranges[range]
             
    def __getitem__(self, idx):
        for range in self.ranges:
            if range.inRange(idx):
                return self.datasets[self.ranges[range]][idx - range.start]
            


class ContrastiveDataset(Dataset):
    def __init__(self,multiLabelDataset:MultiLabelDataset,positives:int = 1,negatives:int = None,random_seed:int = 42) -> None:
        super(ContrastiveDataset,self).__init__()
        self.multiLabelDataset = multiLabelDataset
        self.random_provider = random.Random(random_seed)
        self.possible_indices = list(range(len(multiLabelDataset)))
        while len(self.possible_indices)%2 == 1:
            self.possible_indices.remove(self.possible_indices[self.random_provider.randrange(0, len(self.possible_indices))])


        self.image_pairs = []
        
        dataset_keys = list(self.multiLabelDataset.datasets.keys())
        for index in tqdm(self.possible_indices,"Building Contrastive Pairs"):
            
            sub_dataset =  self.multiLabelDataset.getSubDataset(index)
            #chose random images from the same class
            for i in range(positives):
                positive_range = self.multiLabelDataset.reversed_ranges[sub_dataset]
                other_positive_index = index 
                while other_positive_index == index:
                    other_positive_index = self.random_provider.randrange(positive_range.start, positive_range.stop)
                self.image_pairs.append((index,other_positive_index))
              
            if negatives == None:
                 for key in self.multiLabelDataset.datasets:
                    if key == sub_dataset:
                        continue
                    negative_range = self.multiLabelDataset.reversed_ranges[key]
                    negative_index = self.random_provider.randrange(negative_range.start, negative_range.stop)
                    self.image_pairs.append((index,negative_index))
            else:         
                for i in range(negatives):
                    other_dataset = sub_dataset
                    while other_dataset == sub_dataset:
                        other_dataset = self.random_provider.choice(dataset_keys)
                    
                    negative_range = self.multiLabelDataset.reversed_ranges[other_dataset]
                    negative_index = self.random_provider.randrange(negative_range.start, negative_range.stop)
                    self.image_pairs.append((index,negative_index))
                

            
    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        index1,index2 = self.image_pairs[idx]
        image1,label1 =  self.multiLabelDataset[index1]
        image2,label2 =  self.multiLabelDataset[index2]
        
        return image1,image2,torch.eq(label1, label2).float()
            
              
                
if __name__ == "__main__":
    dataset = MultiLabelDataset(os.path.abspath("./Carparts"),max_size_per_class=100)
    print(len(dataset))
    
    
