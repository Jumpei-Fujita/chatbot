import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorch_transformers import BertTokenizer, BertModel
import time
import matplotlib.pyplot as plt
import seaborn as sns

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def make_mask(n_head, length):
    a = torch.ones(1, n_head, length, length)
    b = torch.triu(a, diagonal=0)
    return b.transpose(2,3)

def make_question(question, tokenizer):
    question = tokenizer.tokenize(question)
    q = tokenizer.convert_tokens_to_ids(question)
    return torch.LongTensor(q).unsqueeze(0)

def make_answer(answer, tokenizer):
    q = '[SEP]' + answer + '[CLS]'
    q = tokenizer.tokenize(q)
    q = tokenizer.convert_tokens_to_ids(q)
    return torch.LongTensor(q).unsqueeze(0)

def get_input(n_head, dataset, data_num, tokenizer, gpu):
    question = dataset['question'][data_num]
    answer = dataset['answer'][data_num]
    
    question = make_question(question, tokenizer)
    answer = make_answer(answer, tokenizer)
    dec_mask = make_mask(n_head, length=len(answer[0]))
    sep = torch.LongTensor([tokenizer.sep_token_id]).unsqueeze(0)
    cls = torch.LongTensor([tokenizer.cls_token_id]).unsqueeze(0)

    if gpu:
        question = question.cuda()
        answer = answer.cuda()
        dec_mask = dec_mask.cuda()
        sep = sep.cuda()
        cls = cls.cuda()
    return question, answer, dec_mask, sep, cls


def train_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer):
    lr_list = np.random.choice(len(train_dataset), (batch_size, ))
    criterion = nn.CrossEntropyLoss()
    loss = 0
    model.train()
    optimizer.zero_grad()
    for i in lr_list:
        question, answer, _, _, _ = get_input(1, train_dataset, i, tokenizer, gpu)
        output = model(question, answer)
        loss_i = criterion(output[0, :, :], answer[0, 1:])
        loss = loss + loss_i / batch_size
        del loss_i, question, answer
    loss.backward()
    optimizer.step()
    loss_item = loss.item()
    del loss
    return loss_item

def validation_i(model, optimizer, batch_size, gpu, validation_dataset, tokenizer):
    lr_list = np.random.choice(len(validation_dataset), (batch_size, ))
    criterion = nn.CrossEntropyLoss()
    loss = 0
    with torch.no_grad():
        model.eval()
        for i in lr_list:
            question, answer, _, _, _ = get_input(1, validation_dataset, i, tokenizer, gpu)
            output = model(question, answer)
            loss_i = criterion(output[0, :, :], answer[0, 1:])
            loss = loss + loss_i / batch_size
            del loss_i, question, answer
        loss_item = loss.item()
        del loss
        return loss_item


def validation_generation(model, validation_dataset, gpu):
    if gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    data_num = np.random.choice(len(validation_dataset))
    question, answer, _, sep, cls  = get_input(1, validation_dataset, data_num, tokenizer, gpu)
    print("~~~~")
    print("question : ", tokenizer.decode(question[0].cpu().numpy()))
    print("gold answer : ", model.word_decode(answer[:, :-1]))
    print("Generated Answer : ", model.generate(question, sep, cls)[0])
    print("~~~~")

def test_generation(model, test_dataset, gpu, data_num, tmp=0.9, top_p=0, top_k=40, tokenizer=tokenizer, viz=False):
    if gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    question, answer, _, sep, cls  = get_input(1, test_dataset, data_num, tokenizer, gpu)
    generated, attention, g_ids = model.generate2(question, sep, cls, tmp, top_p, top_k)
    if viz:
        visualize_attention(attention, question, g_ids)
    
    print("~~~~")
    print("question : ", tokenizer.decode(question[0].cpu().numpy()))
    print("gold answer : ", model.word_decode(answer[:, :-1]))
    print("Generated Answer : ", generated)

def QandA(model, question, gpu, show_Q, tokenizer=tokenizer, viz=False):
    if show_Q:
        print("Q : ", question)
    question = make_question(question, tokenizer)
    sep = torch.LongTensor([tokenizer.sep_token_id]).unsqueeze(0)
    cls = torch.LongTensor([tokenizer.cls_token_id]).unsqueeze(0)

    if gpu:
        model = model.cuda()
        question = question.cuda()
        sep = sep.cuda()
        cls = cls.cuda()
    else:
        model = model.cpu()
    with torch.no_grad():
        answ, attention, answ_ids = model.generate2(question, sep, cls)
        print("A : ", answ)
    if len(answ) <= 1:
        viz = False
    if viz:
        visualize_attention(attention, question, answ_ids)
    return answ

def visualize_attention(attention, question, answer, tokenizer=tokenizer):
    attention = np.transpose(attention)[:, :]

    answer = np.reshape(answer, (-1,))
    answer = tokenizer.convert_ids_to_tokens(answer)[1:]

    question = tokenizer.convert_ids_to_tokens(question.view(-1).detach().cpu().numpy())
    

    df = pd.DataFrame(attention, columns=answer, index=question)
    x = len(answer)
    y = len(question)
    plt.figure(figsize=(x/1.5, y/1.5)) 
    sns.heatmap(df,linewidths=.5, cmap='Blues',xticklabels=1,yticklabels=1) 

def train(model, gpu, lr, batch_size, epochs, train_dataset, validation_dataset, show_generate, tokenizer=tokenizer):
    if gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_history = []
    validation_history = []
    start_time = time.time()
    for epoch in range(epochs):
        loss = train_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer)
        train_history.append(loss)
        loss_validation = validation_i(model, optimizer, batch_size, gpu, validation_dataset, tokenizer)
        validation_history.append(loss_validation)
        if epoch % show_generate == 0:
            print("epochs : ", epoch)
            print("train_loss : ", loss)
            print("validation_loss", loss_validation)
            with torch.no_grad():
                validation_generation(model, validation_dataset, gpu)
            show_time = time.time() - start_time
            print("elapsed time : ", show_time)
            print("=============================================================")
    print("training finished")
    plt.plot(train_history, label="train_loss")
    plt.plot(validation_history, label="validation_loss")
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    return model

    

