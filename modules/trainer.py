import torch
import pytorch_lightning as pl


class Trainer(pl.Trainer):
    """
    pl.Trainer to set Precision and Accelerator automatically
    """
    def __init__(self, *args, **kwargs):
        # bf16 is buggy right now (Pytorch lightning 1.6.3) => use FP16
        precision = 16 if torch.cuda.is_bf16_supported() else 32
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        super().__init__(accelerator=accelerator,precision=precision,benchmark=True,enable_checkpointing=False,*args, **kwargs)