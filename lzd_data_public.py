import numpy
import pandas as pd
import random
import torch
from scipy.stats import beta
from collections import Counter
import matplotlib.pyplot as plt


# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

device = get_device()

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)


def count_diff():
    count = 0
    total = 0
    old_df = pd.read_csv("processed_test.csv")
    new_df = pd.read_csv("processed_test_real_label.csv")

    for index, row in old_df.iterrows():
        total += 1
        if int(row["label"]) != int(new_df.iloc[[index]]["label"]):
            print(row["label"])
            print(new_df.iloc[[index]]["label"])
            count += 1
    print(count * 1.0 / total)

def calculate_new_test():
    df = pd.read_csv("processed_test.csv")
    count = 0
    count_1 = 0
    for index, row in df.iterrows():
        count_1 += 1
        if row["real_label"] != row["supervised"]:
            count += 1
    print(count)
    print(count_1)
    print(count * 1.0 / count_1)

def verification_test():
    orig_test = pd.read_csv("lzd_data_public/full_testset.csv")
    new_test = pd.read_csv("processed_test_real_label.csv")
    count_old = 0
    count_new = 0
    total = 0
    for index, row in orig_test.iterrows():
        total += 1
        if int(row["label"]) != int(new_test.iloc[[index]]["label"]):
            # print(row["label"])
            # print(count_new.iloc[[index]]["label"])
            count_new += 1
    print(count_new * 1.0/total)

def verification_train():
    orig_train = pd.read_csv("lzd_data_public/full_trainset.csv")
    new_train = pd.read_csv("processed_train.csv")
    count_old = 0
    count_new = 0
    total = 0
    for index, row in orig_train.iterrows():
        total += 1
        if int(row["label"]) != int(new_train.iloc[[index]]["label"]):
            # print(row["label"])
            # print(new_train.iloc[[index]]["label"])
            # print(new_train.iloc[[index]]["elapsed_day"])
            # print(new_train.iloc[[index]]["cv_delay_day"])
            count_new += 1
    print(count_new * 1.0/total)

def produce_train():
    df = pd.read_csv("lzd_data_public/full_trainset.csv")

    df["real_label"] = df["label"]
    elapsed_day = []
    cv_delay_day = []
    labels = []

    for index, row in df.iterrows():
        curr_elap = random.randint(0, 3)
        curr_delay = int(numpy.random.exponential(4))
        curr_label = row["real_label"]

        if row["real_label"] == 0:
            curr_delay = None
        else:
            if curr_elap < curr_delay:
                curr_delay = None
                curr_label = 0

        elapsed_day.append(curr_elap)
        cv_delay_day.append(curr_delay)
        labels.append(curr_label)

    df["label"] = labels
    df["elapsed_day"] = elapsed_day
    df["cv_delay_day"] = cv_delay_day
    df["supervised"] = labels
    df.to_csv("processed_train.csv")

def produce_test():
    df = pd.read_csv("lzd_data_public/full_testset.csv")

    df["real_label"] = df["label"]
    elapsed_day = []
    cv_delay_day = []
    labels = []

    for index, row in df.iterrows():
        curr_elap = random.randint(0, 3)
        curr_delay = int(numpy.random.exponential(4))
        curr_label = row["real_label"]

        if row["real_label"] == 0:
            curr_delay = None
        else:
            if curr_elap < curr_delay:
                curr_delay = None
                curr_label = 0

        elapsed_day.append(curr_elap)
        cv_delay_day.append(curr_delay)
        labels.append(curr_label)

    df["label"] = labels
    df["elapsed_day"] = elapsed_day
    df["cv_delay_day"] = cv_delay_day
    df["supervised"] = labels
    df.to_csv("processed_test.csv")
