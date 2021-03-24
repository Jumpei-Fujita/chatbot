import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    
    return logits

def sampling_next_token_pk(temperature, logits, top_k=0, top_p=0.0):


  # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
  logits = logits[0, -1, :] / temperature
  filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

  # Sample from the filtered distribution
  probabilities = F.softmax(filtered_logits, dim=-1)
  
  next_token = torch.multinomial(probabilities, 1)

  return next_token

class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()

        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(768, tokenizer.vocab_size)
    
    def forward(self, question):
        out = self.gpt(question)
        out = self.dropout(out.last_hidden_state)
        out = self.projection(out)
        return out

    def generate(self, question, sep, cls, tmp=0.9, top_k=40, top_p=0, max_len=50):
        with torch.no_grad():
            for i in range(max_len):
                d_out  = self.forward(question)

                d_out = d_out[0, -1].view(1, 1, -1)
                word = sampling_next_token_pk(tmp, d_out, top_k, top_p).view(1, 1)
                if (word == cls) or (word == sep):
                    break
                question = torch.cat((question, word), dim=-1)
        question = question.cpu().view(-1).numpy()
        return tokenizer.decode(question)