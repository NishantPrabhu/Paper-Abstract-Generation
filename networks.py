
""" 
Network definitions
"""

import torch
from transformers import BartForConditionalGeneration 


def get_gpt2_model(freeze_layers=8):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

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


if __name__ == "__main__":

    model = get_gpt2_model(6)