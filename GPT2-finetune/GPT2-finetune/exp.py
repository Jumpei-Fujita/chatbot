import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_input(dataset, data_num, gpu, n_head, generate=False, tokenizer=tokenizer):
    bos = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<"))
    eos = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(">"))
    question = dataset['question'][data_num] + '<'
    answer = dataset['answer'][data_num]
    if not generate:
        question = question + answer + '>'
    question = tokenizer.tokenize(question)
    question = torch.LongTensor([tokenizer.convert_tokens_to_ids(question)])
    sep = torch.LongTensor([bos])
    cls = torch.LongTensor([eos])
    dec_mask = make_mask(n_head, len(question[0]))
    if gpu:
        question = question.cuda()
        sep = sep.cuda()
        cls = cls.cuda()
        dec_mask = dec_mask.cuda()
    return question, sep, cls, dec_mask, answer

def make_mask(n_head, length):
    a = torch.ones(1, n_head, length, length)
    b = torch.triu(a, diagonal=0)
    return b.transpose(2,3)

def train_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer, n_head):
    lr_list = np.random.choice(len(train_dataset), (batch_size, ))
    criterion = nn.CrossEntropyLoss()
    loss = 0
    model.train()
    optimizer.zero_grad()
    for i in lr_list:
        question, sep, cls, dec_mask, answer = get_input(train_dataset, i, gpu, 1)
        output = model(question)
        loss_i = criterion(output[0, :-1, :], question[0, 1:])
        loss = loss + loss_i / batch_size
        del loss_i, question, answer
    loss.backward()
    optimizer.step()
    loss_item = loss.item()
    del loss
    return loss_item

def validation_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer, n_head):
    lr_list = np.random.choice(len(train_dataset), (batch_size, ))
    criterion = nn.CrossEntropyLoss()
    loss = 0
    model.eval()

    for i in lr_list:
        question, sep, cls, dec_mask, answer = get_input(train_dataset, i, gpu, 1)
        output = model(question)
        loss_i = criterion(output[0, :-1, :], question[0, 1:])
        loss = loss + loss_i / batch_size
        del loss_i, question, answer

    loss_item = loss.item()
    del loss
    return loss_item


def validation_generation(model, validation_dataset, tokenizer, gpu):
    if gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    data_num = np.random.choice(len(validation_dataset))
    question, sep, cls, dec_mask, answer = get_input(validation_dataset, data_num, gpu, 1, generate=True, tokenizer=tokenizer)
    print("~~~~")
    print("question : ", tokenizer.decode(question[0].cpu().numpy()).replace("<", ""))
    print("gold answer : ", answer)
    answer = model.generate(question, sep, cls)
    print("Generated Answer : ", answer[answer.find("<"):].replace("<", ""))
    print("~~~~")

def test_generation(model, validation_dataset, data_num, tokenizer, gpu):
    if gpu:
        model = model.cuda()
    else:
        model = model.cpu()

    question, sep, cls, dec_mask, answer = get_input(validation_dataset, data_num, gpu, 1, generate=True, tokenizer=tokenizer)
    print("~~~~")
    print("question : ", tokenizer.decode(question[0].cpu().numpy()).replace("<", ""))
    print("gold answer : ", answer)
    answer = model.generate(question, sep, cls)
    print("Generated Answer : ", answer[answer.find("<"):].replace("<", ""))
    print("~~~~")

def train(model, gpu, lr, batch_size, epochs, train_dataset, validation_dataset, show_generate, layers_num, d_model, n_head, tokenizer=tokenizer):
    if gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_history = []
    validation_history = []
    start_time = time.time()
    for epoch in range(epochs):
        loss = train_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer, n_head)
        train_history.append(loss)
        with torch.no_grad():
            loss_validation = validation_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer, n_head)
        validation_history.append(loss_validation)
        if epoch % show_generate == 0:
            print("epochs : ", epoch)
            print("train_loss : ", loss)
            print("validation_loss", loss_validation)
            with torch.no_grad():
                validation_generation(model, validation_dataset, tokenizer, gpu)
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

def QandA(model, question, gpu, show_Q, tokenizer, dataset, layers_num, d_model):
    if show_Q:
        print("Q : ", question)
    question = tokenizer.tokenize(question + "<")
    question = torch.LongTensor([tokenizer.convert_tokens_to_ids(question)]).view(1, -1)
    if gpu:
        question = question.cuda()
    _, sep, cls, _, _ = get_input(dataset, 0, gpu, 1, True)
    answer = model.generate(question, sep, cls)
    answer = answer[answer.find("<"):].replace("<", "")
    print("A : ", answer)
    return answer
