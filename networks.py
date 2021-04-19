
""" 
Network definitions
"""

import torch
from transformers import GPT2Model 


def get_gpt2_model(freeze_layers=6):
    model = GPT2Model.from_pretrained('gpt2')

    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):        
        if i >= freeze_layers:
            for parameter in m.parameters():
                parameter.requires_grad = True 

    for parameter in model.transformer.ln_f.parameters():        
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():        
        parameter.requires_grad = True

    return model