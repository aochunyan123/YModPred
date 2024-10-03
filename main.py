#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2024.02.05
# @Author : aochunyan
# @Email : acy196707@163.com
# @IDE : PyCharm
# @File : main.py

import os
import argparse
import re
import torch
import pickle
import time
import pandas as pd
import numpy as np
import torch.nn.functional as F
from preprocess import data_process
from model import YModPred_m6A
from model import YModPred

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
residue2idx = pickle.load(open('./data/residue2idx.pkl', 'rb'))

result_folder = 'Results'
if not os.path.exists(result_folder):
        os.makedirs(result_folder)

def model_eval_m6A(test_data, model):
    label_pred = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)
    model.eval()
    with torch.no_grad():
        for batch in test_data:
            inputs = batch.to(device)
            outputs, _ = model(inputs)
            pred_prob_all = F.softmax(outputs, dim=1)
            pred_prob_positive = pred_prob_all[:, 1]
            pred_prob_sort = torch.max(outputs, 1)
            pred_class = pred_prob_sort[1]
            label_pred = torch.cat([label_pred, pred_class.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])
    return label_pred.cpu().numpy(), pred_prob.cpu().numpy()


def model_eval(test_data, model):
    label_pred = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)
    model.eval()
    with torch.no_grad():
        for batch in test_data:
            inputs = batch.to(device)
            outputs = model(inputs)
            pred_prob_all = F.softmax(outputs, dim=1)
            pred_prob_positive = pred_prob_all[:, 1]
            pred_prob_sort = torch.max(outputs, 1)
            pred_class = pred_prob_sort[1]
            label_pred = torch.cat([label_pred, pred_class.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])
    return label_pred.cpu().numpy(), pred_prob.cpu().numpy()


def read_RNA_file(file):
    sequences = []
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    for fasta in records:
        array = fasta.split('\n')
        sequence = re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        sequence = sequence.replace('U', 'T')
        sequences.append(sequence)

    return sequences

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",description="YModPred: a general framework for predicting multiple modifications in saccharomyces")
    parser.add_argument("-i", required=True, default=None, help="input fasta file")
    parser.add_argument("-mod", required=True, help="specify the modification type (e.g., m6A, m1A, m5C, etc.)")
    parser.add_argument("-o", default="Results.csv", help="output a CSV results file")

    args = parser.parse_args()
    time_start = time.time()
    print("Data loading......")

    sequences_list = read_RNA_file(args.i)
    print("Data processing......")
    test_loader = data_process.load_data(sequences_list)
    print("Model loading......")

    models = {
        "m6A": "./data/m6A_model.pkl",
        "m1A": "./data/m1A_model.pkl",
        "m5C": "./data/m5C_model.pkl",
        "m5U": "./data/m5U_model.pkl",
        "m1G": "./data/m1G_model.pkl",
        "m2G": "./data/m2G_model.pkl",
        "m2_2G": "./data/m2_2G_model.pkl",
        "f5C": "./data/f5c_model.pkl",
        "D": "./data/D_model.pkl",
        "pse": "./data/pse_model.pkl"}

    if args.mod not in models:
        print(
            f"Error: Unsupported modification type '{args.mod}'. Supported types are: {', '.join(models.keys())}")
        exit(1)

    model_path = models[args.mod]
    print(f"Loading model for {args.mod} from {model_path}......")

    if args.mod == "m6A":
        model = YModPred_m6A.YModPred().to(device)
    else:
        model = YModPred.YModPred().to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    print("Predicting......")
    if args.mod == "m6A":
        y_pred, y_pred_prob = model_eval_m6A(test_loader, model)
    else:
        y_pred, y_pred_prob = model_eval(test_loader, model)


    for i, seq in enumerate(sequences_list):
        sequences_list[i] = ''.join(seq)

    results = pd.DataFrame(np.zeros([len(y_pred), 4]), columns=["Seq_ID", "Sequences", "Prediction", "Confidence"])
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            y_prob = str(round(y_pred_prob[i] * 100, 2)) + "%"
            results.iloc[i, :] = [round(i + 1), sequences_list[i], "Modification Site", y_prob]
        else:
            y_prob = str(round((1-y_pred_prob[i]) * 100, 2)) + "%"
            results.iloc[i, :] = [round(i + 1), sequences_list[i], "Non Modification Site", y_prob]
    os.chdir("Results")
    results.to_csv(args.o, index=False)
    print("job finished!")

    time_end = time.time()
    print('Total time cost', time_end - time_start, 'seconds')
