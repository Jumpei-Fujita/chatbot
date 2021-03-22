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

class Attention(nn.Module):
    def __init__(self, d_model, layers_num, dropout=0.1):
        super(Attention, self).__init__()
        
        self.w_d = nn.Linear(d_model*layers_num, d_model, bias=False)
        self.w_e = nn.Linear(2 * d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.layers_num = layers_num
        self.Softmax = nn.Softmax(dim=-1)


    def forward(self, h, question_hidden):

        h = h.contiguous().view(1, 1, self.layers_num*self.d_model)
        h = self.w_d(h)
        h = self.layer_norm(self.dropout(h))

        q = self.w_e(question_hidden)
        q = self.layer_norm(self.dropout(q))

        #h : [1, 1, d_model], q : [1, len, d_model]
        a = torch.matmul(q / (self.d_model**0.5), h.transpose(1, 2))
        #a : [1, len, 1]

        a = a.transpose(1, 2)
        #a : [1, 1, len]
        a = self.Softmax(a)

        c = torch.matmul(a, q)

        return a, c

class LSTMDecoder(nn.Module):
    def __init__(self, embedding, word_num=tokenizer.vocab_size, d_model=256, layers_num=2, dropout=0.1):
        super(LSTMDecoder, self).__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, d_model, layers_num, batch_first=True, bidirectional=False)
        self.attention = Attention(d_model, layers_num, dropout)
        self.projection = nn.Linear(d_model, word_num)

    def forward(self, dec_input, question_hidden, h, c):

        d_embedding = self.embedding(dec_input)
        d = self.dropout(d_embedding)
        d = self.layer_norm(d)
        d_hidden, (h, c) = self.lstm(d, (h, c))
        a, context = self.attention(h, question_hidden)
        d_out = self.projection(context)
        
        return d_out, (h, c), a
    
class LSTMEncoderDecoder_Attention(nn.Module):
    def __init__(self, word_num=tokenizer.vocab_size, d_model=512, layers_num=2, dropout=0.1):
        super(LSTMEncoderDecoder_Attention, self).__init__()

        self.encoder = LSTMEncoder(word_num, d_model, layers_num, dropout)

        self.embedding = nn.Embedding(word_num, d_model)
        
        self.decoder = LSTMDecoder(self.embedding, word_num, d_model, layers_num, dropout)


    def forward(self, question, answer):
        question_hidden, (h, c) = self.encoder(question)

        output_list = []
        for i in range(len(answer[0]) - 1):
            dec_input = answer[:, :i+1].contiguous().view(1, i+1)

            d_out, (h, c), a = self.decoder(dec_input, question_hidden, h, c)

            output_list.append(d_out)            
        return torch.cat(output_list, dim=1)
    
    def generate(self, question, sep, cls, max_num=50):

            question_hidden, (h, c) = self.encoder(question)
            dec_output_list = []
            attention_list = []
            dec_output_list.append(sep)
            for i in range(max_num):
                if i != 0:
                    dec_input = torch.cat(dec_output_list, dim=-1)
                else:
                    dec_input = dec_output_list[0]
                d_out, (h, c), a = self.decoder(dec_input, question_hidden, h, c)
                word = torch.argmax(d_out[:, -1], dim=-1).unsqueeze(0)
                if word == cls:
                    break
                dec_output_list.append(word)
                attention_list.append(a.cpu())
            
            if len(attention_list) > 1:
                att = torch.cat(attention_list, dim=1).squeeze(0).detach().numpy()
            if len(attention_list) == 1 :
                att = attention_list[0]
            if len(attention_list) == 0:
                att = torch.zeros(1, 1)
                
            return self.word_decode(dec_input), att, dec_input.detach().cpu().numpy()
            
    def word_decode(self, dec_input):
            
            dec_input = dec_input.cpu().numpy()
            return tokenizer.decode(dec_input[0, 1:])
    
    def generate2(self, question, sep, cls, tmp=0.9, top_p=0, top_k=40, max_num=50):

            question_hidden, (h, c) = self.encoder(question)

            dec_output_list = []
            attention_list = []
            dec_output_list.append(sep)
            for i in range(max_num):
                if i != 0:
                    dec_input = torch.cat(dec_output_list, dim=-1)
                else:
                    dec_input = dec_output_list[0]
                d_out, (h, c), a = self.decoder(dec_input, question_hidden, h, c)
                word = sampling_next_token_pk(tmp, d_out[:, -1].unsqueeze(0), top_k, top_p).unsqueeze(0)
                if word == cls:
                    break
                dec_output_list.append(word)
                attention_list.append(a.cpu())
            if len(attention_list) > 1:
                att = torch.cat(attention_list, dim=1).squeeze(0).detach().numpy()
            if len(attention_list) == 1 :
                att = attention_list[0]
            if len(attention_list) == 0:
                att = torch.zeros(1, 1)
                
            return self.word_decode(dec_input), att, dec_input.detach().cpu().numpy()
            

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



