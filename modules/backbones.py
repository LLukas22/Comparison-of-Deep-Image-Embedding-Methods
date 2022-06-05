import torch
import torchvision.models as models
from torchinfo import summary
import timm 
from pprint import pprint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_EMBEDDING = 256

class EmbeddingNetwork(torch.nn.Module):
    def __init__(self,embedding_size:int=DEFAULT_EMBEDDING) -> None:
        super(EmbeddingNetwork, self).__init__()
        self.embedding_size = embedding_size

    def build_fc(self,feature_input_size:int):
        return torch.nn.Sequential(
            torch.nn.Linear( feature_input_size,  self.embedding_size*2),
            torch.nn.BatchNorm1d( self.embedding_size*2),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear( self.embedding_size*2,  self.embedding_size),
        )
    
class ResNet50(EmbeddingNetwork):
    def __init__(self,pretrained:bool=True,freeze:bool=True,embedding_size: int = DEFAULT_EMBEDDING) -> None:
        super().__init__(embedding_size)
        self.model = models.resnet50(pretrained=pretrained)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = self.build_fc(self.model.fc.in_features)
        
    def forward(self,x):
        return self.model(x)
        
class DenseNet169(EmbeddingNetwork):
    def __init__(self,pretrained:bool=True, freeze:bool=True, embedding_size: int = DEFAULT_EMBEDDING) -> None:
        super().__init__(embedding_size)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=pretrained)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier = self.build_fc(self.model.classifier.in_features)
        
    def forward(self,x):
        return self.model(x)
    
    
class EfficientNetV2_L(EmbeddingNetwork):
    def __init__(self,pretrained:bool=True, freeze:bool=True, embedding_size: int = DEFAULT_EMBEDDING) -> None:
        super().__init__(embedding_size)
        
        self.model = timm.create_model('tf_efficientnetv2_l_in21k', pretrained=pretrained)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier = self.build_fc(self.model.classifier.in_features)
        
    def forward(self,x):
        return self.model(x)
    
    
class MobilNetV3(EmbeddingNetwork):
    def __init__(self,pretrained:bool=True, freeze:bool=True, embedding_size: int = DEFAULT_EMBEDDING) -> None:
        super().__init__(embedding_size)
        self.model = models.mobilenet_v3_large(pretrained=pretrained)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier = self.build_fc(self.model.classifier[0].in_features)
        
    def forward(self,x):
        return self.model(x)
    
    
class Swin(EmbeddingNetwork):
    def __init__(self,pretrained:bool=True, freeze:bool=True, embedding_size: int = DEFAULT_EMBEDDING) -> None:
        super().__init__(embedding_size)
        
        self.model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=pretrained)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.head = self.build_fc(self.model.head.in_features)
        self.pooling = torch.nn.AdaptiveAvgPool2d((224,224))
        
    def forward(self,x):
        x = self.pooling(x)
        return self.model(x)
    
    
class ViT(EmbeddingNetwork):
    def __init__(self,pretrained:bool=True, freeze:bool=True, embedding_size: int = DEFAULT_EMBEDDING) -> None:
        super().__init__(embedding_size)
        
        self.model = timm.create_model('vit_small_patch16_224_in21k', pretrained=pretrained)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.head = self.build_fc(self.model.head.in_features)
        self.pooling = torch.nn.AdaptiveAvgPool2d((224,224))
        
    def forward(self,x):
        x = self.pooling(x)
        return self.model(x)
    
if __name__ == "__main__":
    summary(ViT().to(DEVICE),(1, 3, 64, 64))
    # avail_pretrained_models = timm.list_models(pretrained=True)
    # pprint(avail_pretrained_models)
