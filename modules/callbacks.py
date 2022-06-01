from typing import Any, Dict, List, Optional, Type

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pandas as pd


class LossTracker(Callback):

    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        loss:torch.Tensor = pl_module.train_loss
        self.train_loss.append(loss.cpu().detach().numpy().item())

    
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        loss:torch.Tensor = pl_module.val_loss
        self.val_loss.append(loss.cpu().detach().numpy().item())
        
    def save(self,dir:str)->None:
        train_df = pd.DataFrame(self.train_loss,columns=["loss"])
        train_df.to_csv(os.path.join(dir,'train_loss.csv'),index=False)
        val_df = pd.DataFrame(self.val_loss,columns=["loss"])
        val_df.to_csv(os.path.join(dir,'val_loss.csv'),index=False)
        
        
class ModelSaver(Callback):

    def __init__(self,dir:str)->None:
        self.dir = os.path.join(dir,"checkpoints")
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir,exist_ok=True)
            
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        torch.save(pl_module.model.state_dict(), os.path.join(self.dir, f"epoch_{trainer.current_epoch}.pth"))
