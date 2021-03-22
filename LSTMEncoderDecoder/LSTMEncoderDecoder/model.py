import torch
import torch.nn as nn
import numpy as np
from pytorch_transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    def __init__(self, word_num=tokenizer.vocab_size, d_model=256, layers_num=2, dropout=0.1):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(word_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, d_model, layers_num, batch_first=True, bidirectional=True)

        self.layers_num = layers_num

    def forward(self, question):
        question_embedding = self.embedding(question)
        question = self.dropout(question_embedding)
        question = self.layer_norm(question)

        question_hidden, (h, c) = self.lstm(question_embedding)
        h = h[:self.layers_num] + h[self.layers_num:]
        c = c[:self.layers_num] + c[self.layers_num:]
        return question_hidden, (h, c)

class LSTMDecoder(nn.Module):
    def __init__(self, embedding, word_num=tokenizer.vocab_size, d_model=256, layers_num=2, dropout=0.1):
        super(LSTMDecoder, self).__init__()
        self.embedding = embedding
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

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, word_num=tokenizer.vocab_size, d_model=256, layers_num=2, dropout=0.1):
        super(LSTMEncoderDecoder, self).__init__()

        self.encoder = LSTMEncoder(word_num, d_model, layers_num, dropout)
        
        self.decoder = LSTMDecoder(self.encoder.embedding, word_num, d_model, layers_num, dropout)


    def forward(self, question, answer):
        question_hidden, (h, c) = self.encoder(question)
        d_out, (h, c) = self.decoder(answer, h, c)
        return d_out
    
    def generate(self, question, sep, cls, max_num=50):

            question_hidden, (h, c) = self.encoder(question)
            dec_output_list = []
            dec_output_list.append(sep)
            for i in range(max_num):
                if i != 0:
                    dec_input = torch.cat(dec_output_list, dim=-1)
                else:
                    dec_input = dec_output_list[0]
                d_out, (h, c) = self.decoder(dec_input, h, c)
                word = torch.argmax(d_out[:, -1], dim=-1).unsqueeze(0)
                if word == cls:
                    break
                dec_output_list.append(word)
            return self.word_decode(dec_input)
            
    def word_decode(self, dec_input):
            
            dec_input = dec_input.cpu().numpy()
            return tokenizer.decode(dec_input[0, 1:])
    
    def generate2(self, question, sep, cls, tmp=0.9, top_p=0, top_k=40, max_num=50):

            question_hidden, (h, c) = self.encoder(question)
            dec_output_list = []
            dec_output_list.append(sep)
            for i in range(max_num):
                if i != 0:
                    dec_input = torch.cat(dec_output_list, dim=-1)
                else:
                    dec_input = dec_output_list[0]
                d_out, (h, c) = self.decoder(dec_input, h, c)
                word = sampling_next_token_pk(tmp, d_out[:, -1].unsqueeze(0), top_k, top_p).unsqueeze(0)
                if word == cls:
                    break
                dec_output_list.append(word)
            return self.word_decode(dec_input)
            

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


