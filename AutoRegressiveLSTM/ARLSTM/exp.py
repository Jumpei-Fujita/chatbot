from pytorch_transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time



def get_input(dataset, data_num, gpu, layers_num, d_model, generate=False, tokenizer=tokenizer):
    question = dataset['question'][data_num] + '[PAD]'
    answer = dataset['answer'][data_num]
    if not generate:
        question = question + answer + '[CLS]'
    question = tokenizer.tokenize(question)
    question = torch.LongTensor([tokenizer.convert_tokens_to_ids(question)])
    sep = torch.LongTensor([tokenizer.pad_token_id]).unsqueeze(0)
    cls = torch.LongTensor([tokenizer.cls_token_id]).unsqueeze(0)
    h = torch.zeros(layers_num, 1, d_model)
    c = h
    if gpu:
        question = question.cuda()
        sep = sep.cuda()
        cls = cls.cuda()
        h = h.cuda()
        c = c.cuda()
    return question, sep, cls, answer, h, c

def train_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer, layers_num, d_model):
    lr_list = np.random.choice(len(train_dataset), (batch_size, ))
    criterion = nn.CrossEntropyLoss()
    loss = 0
    model.train()
    optimizer.zero_grad()
    for i in lr_list:
        question, sep, cls, answer, h, c = get_input(train_dataset, i, gpu, layers_num, d_model)
        output, _ = model(question, h, c)
        loss_i = criterion(output[0, :-1, :], question[0, 1:])
        loss = loss + loss_i / batch_size
        del loss_i, question, answer
    loss.backward()
    optimizer.step()
    loss_item = loss.item()
    del loss
    return loss_item

def validation_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer, layers_num, d_model):
    lr_list = np.random.choice(len(train_dataset), (batch_size, ))
    criterion = nn.CrossEntropyLoss()
    loss = 0
    model.train()
    optimizer.zero_grad()
    for i in lr_list:
        question, sep, cls, answer, h, c = get_input(train_dataset, i, gpu, layers_num, d_model)
        output, _ = model(question, h, c)
        loss_i = criterion(output[0, :-1, :], question[0, 1:])
        loss = loss + loss_i / batch_size
        del loss_i, question, answer
    loss.backward()
    optimizer.step()
    loss_item = loss.item()
    del loss
    return loss_item

def validation_generation(model, validation_dataset, layers_num, d_model, tokenizer, gpu):
    if gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    data_num = np.random.choice(len(validation_dataset))
    question, sep, cls, answer, h, c = get_input(validation_dataset, data_num, gpu, layers_num, d_model, generate=True, tokenizer=tokenizer)
    print("~~~~")
    print("question : ", tokenizer.decode(question[0].cpu().numpy()).replace("[PAD]", ""))
    print("gold answer : ", answer)
    answer = model.generate(question, h, c, sep, cls)
    print("Generated Answer : ", answer[answer.find("[PAD]"):].replace("[PAD]", ""))
    print("~~~~")

def test_generation(model, test_dataset, data_num, layers_num, d_model, tokenizer, gpu):
    if gpu:
        model = model.cuda()
    else:
        model = model.cpu()

    question, sep, cls, answer, h, c = get_input(test_dataset, data_num, gpu, layers_num, d_model, generate=True, tokenizer=tokenizer)
    print("~~~~")
    print("question : ", tokenizer.decode(question[0].cpu().numpy()).replace("[PAD]", ""))
    print("gold answer : ", answer)
    answer = model.generate(question, h, c, sep, cls)
    print("Generated Answer : ", answer[answer.find("[PAD]"):].replace("[PAD]", ""))
    print("~~~~")

def train(model, gpu, lr, batch_size, epochs, train_dataset, validation_dataset, show_generate, layers_num, d_model, tokenizer=tokenizer):
    if gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_history = []
    validation_history = []
    start_time = time.time()
    for epoch in range(epochs):
        loss = train_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer, layers_num, d_model)
        train_history.append(loss)
        loss_validation = validation_i(model, optimizer, batch_size, gpu, train_dataset, tokenizer, layers_num, d_model)
        validation_history.append(loss_validation)
        if epoch % show_generate == 0:
            print("train_loss : ", loss)
            print("validation_loss", loss_validation)
            with torch.no_grad():
                validation_generation(model, validation_dataset, layers_num, d_model, tokenizer, gpu)
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
    question = tokenizer.tokenize(question + "[PAD]")
    question = torch.LongTensor([tokenizer.convert_tokens_to_ids(question)]).view(1, -1)
    if gpu:
        question = question.cuda()
    _, sep, cls, _, h, c = get_input(dataset, 0, gpu, layers_num, d_model, generate=True)
    answer = model.generate(question, h, c, sep, cls)
    answer = answer[answer.find("[PAD]"):].replace("[PAD]", "")
    print("A : ", answer)
    return answer