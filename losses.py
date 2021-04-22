
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
        vocab_size = output.size(-1)
        output = output.contiguous().view(-1, vocab_size)
        target = target.contiguous().view(-1,)
        loss = F.cross_entropy(output, target, reduction="mean")
        preds = output.argmax(dim=-1)
        acc = preds.eq(target.view_as(preds)).sum().item() / preds.size(0)
        return loss, acc
