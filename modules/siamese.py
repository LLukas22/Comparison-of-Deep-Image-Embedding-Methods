import torch
import pytorch_lightning as pl
from modules.losses import ContrastiveLoss
#see https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942

class SiameseNetwork(pl.LightningModule):
    """
    Wrapps a embedding model into a siamese network
    """
    def __init__(self,model:torch.nn.Module) -> None:
        super(SiameseNetwork, self).__init__()
        self.model = model
        self.criterion = ContrastiveLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x1,x2,y = batch
        y_hat1 = self(x1)
        y_hat2 = self(x2)
        loss = self.criterion(y_hat1,y_hat2, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x1,x2,y = batch
        y_hat1 = self(x1)
        y_hat2 = self(x2)
        val_loss = self.criterion(y_hat1,y_hat2, y)
        self.log("val_loss", val_loss)