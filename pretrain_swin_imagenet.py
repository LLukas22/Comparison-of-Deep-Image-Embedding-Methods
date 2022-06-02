import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import RandAugment
from modules.losses import SupConLoss
from modules.backbones import Swin
from modules.datasets import MultiLabelDataset,AugmentingDataset
from modules.siamese import ContrastiveNetwork
from modules.callbacks import LossTracker,ModelSaver
from modules.trainer import Trainer
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*") # we want to run Single-Core in the Notebook -> Ignore this warning
warnings.simplefilter(action='ignore', category=FutureWarning)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./runs/zero_shot"

#Sadly this  has to be in a normal *.py file because of the way multiprocessing works in notebooks
if __name__ == "__main__":
    
    dataset = MultiLabelDataset(
        "./tiny-imagenet-200/train",
        transform=None,
        )


    BATCH_SIZE = 128

    augmented_dataset = AugmentingDataset(dataset,RandAugment(),transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),factor=3)
    train_loader = DataLoader(augmented_dataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True,num_workers=10,prefetch_factor=10)

    #Train 1-Epoch on augmented Tiny-ImageNet

    siamese_model = ContrastiveNetwork(Swin(pretrained=True,freeze=False),SupConLoss())

    model_dir = os.path.join(SAVE_DIR,"tiny_imagenet")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir,exist_ok=True)
        lossTracker = LossTracker()
        modelSaver = ModelSaver(model_dir)
        trainer = Trainer(max_epochs=1,callbacks=[lossTracker,modelSaver])
        trainer.fit(model=siamese_model, train_dataloaders=train_loader)
        lossTracker.save(model_dir)