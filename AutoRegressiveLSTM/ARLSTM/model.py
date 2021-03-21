from pytorch_transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

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

class LSTMDecoder(nn.Module):
    def __init__(self, word_num=tokenizer.vocab_size, d_model=256, layers_num=2, dropout=0.1):
        super(LSTMDecoder, self).__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(word_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, d_model, layers_num, batch_first=True, bidirectional=False)
        self.projection = nn.Linear(d_model, word_num)

    def forward(self, dec_input, h, c):
        d_embedding = self.embedding(dec_input)
        d = self.dropout(d_embedding)
        d = self.layer_norm(d)
        d_hidden, (h, c) = self.lstm(d, (h, c))
        d_out = self.projection(d_hidden)
        
        return d_out, (h, c)
    
    def generate(self, question, h, c, sep, cls, tmp=0.9, top_k=40, top_p=0, max_len=50):
        with torch.no_grad():
            for i in range(max_len):
                d_out, (h, c) = self.forward(question, h, c)
                d_out = d_out[0, -1].view(1, 1, -1)
                word = sampling_next_token_pk(tmp, d_out, top_k, top_p).view(1, 1)
                if (word == cls) or (word == sep):
                    break
                question = torch.cat((question, word), dim=-1)
        question = question.cpu().view(-1).numpy()
        return tokenizer.decode(question)


