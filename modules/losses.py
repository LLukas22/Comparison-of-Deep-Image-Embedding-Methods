import torch
from torch import nn
from torch.nn import functional as F
from pytorch_metric_learning import losses

class  SupervisedLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(SupervisedLoss, self).__init__()
        self.loss = None
        
    def forward(self, embeddings, labels):
        return self.loss(embeddings, labels)
        
class UnsupervisedLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(UnsupervisedLoss, self).__init__()
 
        
class ContrastiveLoss(SupervisedLoss):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self,pos_margin=0.0, neg_margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.loss = losses.ContrastiveLoss(pos_margin=pos_margin,neg_margin=neg_margin)

     
class TripletLoss(SupervisedLoss):
    """
    Triplet loss function.
    """
    def __init__(self, margin=0.05,
                        swap=False,
                        smooth_loss=False,
                        triplets_per_anchor="all"):
        super(TripletLoss, self).__init__()
        self.loss = losses.TripletMarginLoss(margin=margin,
                                                swap=swap,
                                                smooth_loss=smooth_loss,
                                                triplets_per_anchor=triplets_per_anchor)

class  SupConLoss(SupervisedLoss):
    """
        Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self,temperature=0.1) -> None:
        super(SupConLoss, self).__init__()
        self.loss = losses.SupConLoss(temperature=temperature)

    def forward(self,features,labels):
        return self.loss(features, labels)
    
    
class SNRLoss(SupervisedLoss):
    """
    SignalToNoiseRatioContrastiveLoss see: http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    """
    def __init__(self,pos_margin=0, neg_margin=1) -> None:
        super(SNRLoss, self).__init__()
        self.loss = losses.SignalToNoiseRatioContrastiveLoss(pos_margin=pos_margin,neg_margin=neg_margin)
        
class NTXentLoss(SupervisedLoss):
    """
    NTXentLoss see: https://arxiv.org/pdf/1807.03748.pdf or https://arxiv.org/pdf/2002.05709.pdf
    """
    def __init__(self,temperature=0.5) -> None:
        super(NTXentLoss, self).__init__()
        self.loss = losses.NTXentLoss(temperature=temperature)