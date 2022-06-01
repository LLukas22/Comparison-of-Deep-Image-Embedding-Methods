import torch
import pytorch_lightning as pl

#see https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942

class ContrastiveNetwork(pl.LightningModule):
    """
    Wrapps a embedding model into a siamese network
    """
    def __init__(self,model:torch.nn.Module,criterion:torch.nn.Module) -> None:
        super(ContrastiveNetwork, self).__init__()
        self.model = model
        self.criterion = criterion
        self.train_loss = None
        self.val_loss = None
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        img, y = batch
        y_hat = self(img)
        self.train_loss = self.criterion(y_hat, y.flatten())
        return self.train_loss
    
    def validation_step(self, batch, batch_idx):
        img, y = batch
        y_hat = self(img)
        self.val_loss = self.criterion(y_hat, y.flatten())
        self.log_dict({'val_loss': self.val_loss})
