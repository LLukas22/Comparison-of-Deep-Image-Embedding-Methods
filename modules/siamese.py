from torch import nn

#see https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942

class SiameseNetwork(nn.Module):
    """
    Wrapps a embedding model into a siamese network
    """
    def __init__(self,model:nn.Module) -> None:
        super(SiameseNetwork, self).__init__()
        self.model = model
        
    def single_forward(self,x):
        """
        the normal forward function of the embedding model
        """
        return self.model(x)
    
    def forward(self, x1, x2):
        """
        the forward function of the siamese network
        """
        y1 = self.single_forward(x1)
        y2 = self.single_forward(x2)
        return y1, y2