import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from parameters import *
from function import *

def ann_train(device,train_dataloader,test_dataloader,model):
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    model.to(device)
    # opt = optim.Adam(model.parameters(), lr=0.000172)
    opt = optim.Adam(model.parameters(), lr=0.000172, weight_decay=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        true_labels = []
        predicted_labels = []
        for inputs, labels in tqdm(train_dataloader):
            inputs = (inputs.reshape(-1,1,time_step,num_channel)).to(device)
            labels = labels.to(device)
            opt.zero_grad()
            outputs = (model(inputs)).to(device)
            # outputs = torch.sum(out,dim=1)
            # labels = labels.to(torch.long)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            loss.backward()
            opt.step()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())
            correct_predictions += torch.sum(predicted == labels).item()
        cm = confusion_matrix(true_labels, predicted_labels)
        train_accuracy = correct_predictions / len(predicted_labels)
        print("Train Accuracy:", train_accuracy)
        print("Train Confusion Matrix:")
        print(cm)
        # 计算测试集的损失和准确率
        test_loss = 0.0
        test_correct_predictions = 0
        model.eval()
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.reshape(-1,1,time_step,num_channel).to(device)
                # inputs[inputs == -1] = 2
                labels = labels.to(device)
                outputs = model(inputs).to(device)
                # outputs = torch.sum(outputs,dim=1)
                labels = labels.to(torch.long)
                loss = criterion(outputs, labels).to(device)
                _, predicted = torch.max(outputs, 1)
                # _, label = torch.max(labels, 1)
                test_loss += loss.item()
                test_correct_predictions += torch.sum(predicted == labels).item()
                true_labels.extend(labels.tolist())
                predicted_labels.extend(predicted.tolist())
        cm = confusion_matrix(true_labels, predicted_labels)
        print("Confusion Matrix:")
        print(cm)
        # 输出训练集和测试集的损失和准确率
        train_loss = total_loss / len(train_dataloader)
        
        test_loss /= len(test_dataloader)
        test_accuracy = test_correct_predictions / len(test_dataloader.dataset)
        print("Epoch", epoch+1)
        # print("Train Set:")
        # print("Loss:", train_loss)
        
        print("Test Set:")
        print("Loss:", test_loss)
        print_colorful_text(f"Accuracy:{test_accuracy}", 'yellow')
        model.eval()
        
        
def snn_train_spike(device, train_dataloader, test_dataloader, model):
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    model.to(device)
    print('device:',device)
    opt = optim.Adam(model.parameters().astorch(), lr=0.000572)
    # scheduler = lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    losslist = []
    accuracy = []
    f1s = []
    precision = []
    recall = []
    cmlist= []
    for epoch in range(2000):
        # scheduler.step()
        train_preds = []
        train_targets = []
        sum_loss = 0.0
        for batch, target in tqdm(train_dataloader):
            batch = batch.to(torch.float32).to(device)
            # target_loss = (torch.tensor(encode_labels(target,Nout,thr_out+0.2))).float().to(device)
            target_loss = target.to(device)
            model.reset_state()
            opt.zero_grad()
            out_model, _, rec = model(batch, record=True)
            out = torch.sum(out_model,dim=1)
            loss = criterion(out, target_loss)
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = out.argmax(1).detach().to(device)
                train_preds += pred.detach().cpu().numpy().tolist()
                train_targets += target.detach().cpu().numpy().tolist()
                sum_loss += loss.item() / len(train_dataloader)

        sum_f1 = f1_score(train_targets, train_preds, average="macro")
        _, train_precision, train_recall, _ = precision_recall_fscore_support(
            train_targets, train_preds, labels=np.arange(8)
        )
        train_accuracy = accuracy_score(train_targets, train_preds)

        print(f"Train Epoch = {epoch+1}, Loss = {sum_loss}, F1 Score = {sum_f1}")
        print(f"Train Precision = {train_precision}, Recall = {train_recall}")
        print(f"Train Accuracy = {train_accuracy}")

        test_preds = []
        test_targets = []
        test_loss = 0.0
        for batch, target in tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.to(torch.float32).to(device)
                model.reset_state()
                out_model, _, rec = model(batch, record=True)
                out = torch.sum(out_model,dim=1)
                pred = out.argmax(1).detach().to(device)
                test_preds += pred.detach().cpu().numpy().tolist()
                test_targets += target.detach().cpu().numpy().tolist()
 
        f1 = f1_score(test_targets, test_preds, average="macro")
        _, test_precision, test_recall, _ = precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(8)
        )
        test_accuracy = accuracy_score(test_targets, test_preds)
        cm = confusion_matrix(test_targets, test_preds)
        # losslist.append(test_loss)
        f1s.append(f1)
        precision.append(test_precision)
        recall.append(test_recall)
        accuracy.append(test_accuracy)
        cmlist.append(cm)
        print(f"F1 Score = {f1}")
        print(f"Val Precision = {test_precision}, Recall = {test_recall}")
        print_colorful_text(f"Accuracy:{test_accuracy}", 'yellow')
        print("Confusion Matrix:")
        print(cm)
        if test_accuracy > 0.927:
            model.save(f'models/modelmix_spike_{epoch}_{test_accuracy}.pth')
            print('模型已保存')
            # np.save('train_data_record/loss.npy', losslist)
            # np.save('train_data_record/f1s.npy', f1s)
            # np.save('train_data_record/precision.npy', precision)
            # np.save('train_data_record/recall.npy', recall)
            # np.save('train_data_record/accuracy.npy', accuracy)
            # np.save('train_data_record/cm.npy', cmlist)
            print('训练已完成，训练参数已保存') 
            break
        
def snn_train_spike_mix(device, train_dataloader, test_dataloader, model1, model2):
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    model1.to(device)
    model2.to(device)
    print('device:',device)
    opt1 = optim.Adam(model1.parameters().astorch(), lr=0.000172)
    opt2 = optim.Adam(model2.parameters().astorch(), lr=0.000172)
    # scheduler = lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    losslist = []
    accuracy = []
    f1s = []
    precision = []
    recall = []
    cmlist= []
    for epoch in range(2000):
        # scheduler.step()
        train_preds = []
        train_targets = []
        sum_loss = 0.0
        for batch, target in tqdm(train_dataloader):
            batch = batch.to(torch.float32).to(device)
            # target_loss = (torch.tensor(encode_labels(target,Nout,thr_out+0.2))).float().to(device)
            target_loss = target.to(device)
            model1.reset_state()
            opt1.zero_grad()
            opt2.zero_grad()
            
            out_model, _, rec = model1(batch, record=True)
            out_model, _, rec = model2(out_model, record=True)
            out = torch.sum(out_model,dim=1)
            loss = criterion(out, target_loss)
            loss.backward()
            opt1.step()
            opt2.step()

            with torch.no_grad():
                pred = out.argmax(1).detach().to(device)
                train_preds += pred.detach().cpu().numpy().tolist()
                train_targets += target.detach().cpu().numpy().tolist()
                sum_loss += loss.item() / len(train_dataloader)

        sum_f1 = f1_score(train_targets, train_preds, average="macro")
        _, train_precision, train_recall, _ = precision_recall_fscore_support(
            train_targets, train_preds, labels=np.arange(2)
        )
        train_accuracy = accuracy_score(train_targets, train_preds)

        print(f"Train Epoch = {epoch+1}, Loss = {sum_loss}, F1 Score = {sum_f1}")
        print(f"Train Precision = {train_precision}, Recall = {train_recall}")
        print(f"Train Accuracy = {train_accuracy}")

        test_preds = []
        test_targets = []
        test_loss = 0.0
        for batch, target in tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.to(torch.float32).to(device)
                model1.reset_state()
                model2.reset_state()
                out_model, _, rec = model1(batch, record=True)
                out_model, _, rec = model2(out_model, record=True)
                out = torch.sum(out_model,dim=1)
                pred = out.argmax(1).detach().to(device)
                test_preds += pred.detach().cpu().numpy().tolist()
                test_targets += target.detach().cpu().numpy().tolist()
 
        f1 = f1_score(test_targets, test_preds, average="macro")
        _, test_precision, test_recall, _ = precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(2)
        )
        test_accuracy = accuracy_score(test_targets, test_preds)
        cm = confusion_matrix(test_targets, test_preds)
        # losslist.append(test_loss)
        f1s.append(f1)
        precision.append(test_precision)
        recall.append(test_recall)
        accuracy.append(test_accuracy)
        cmlist.append(cm)
        print(f"F1 Score = {f1}")
        print(f"Val Precision = {test_precision}, Recall = {test_recall}")
        print_colorful_text(f"Accuracy:{test_accuracy}", 'yellow')
        print("Confusion Matrix:")
        print(cm)