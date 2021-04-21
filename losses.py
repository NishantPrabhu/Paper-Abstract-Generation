
""" 
Loss function for decoder
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class GenerationLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """ 
        output: (bs, seqlen, vocab_size) 
        target: (bs, seqlen)
        """
        assert output.size(1) == target.size(1), f"Output has size {output.size()} and target has size {target.size()}"
        bs, seqlen = target.size()
        output = output.view(bs * seqlen, -1)
        target = target.view(bs * seqlen,)
        loss = F.cross_entropy(output, target, reduce="mean")
        return loss
