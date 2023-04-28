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


def examine_Twins():
    print("examine_Twins")
    df = pd.read_csv("TWINS/TwinTrain.csv")
    count_y = 0
    count_row = 0
    count_a= 0
    for index, row in df.iterrows():
        count_row += 1
        if row["label"] == 1:
            count_y += 1
        if row["A"] == 1:
            count_a += 1
    print("Train Count positive percentage")
    print(count_y * 1.0 / count_row)

    print("Train Count treatment percentage")
    print(count_a * 1.0 / count_row)

def produce_Twins_train_beta():
    df = pd.read_csv("TWINS/TwinTrain.csv")

    df["real_label"] = df["label"]
    elapsed_day = []
    cv_delay_day = []
    labels = []

    real_delay_time = []
    sim_delay_time = []

    for index, row in df.iterrows():
        # user related features extracted
        user_feat = row[10:100]
        user_feat = df_to_tensor(user_feat)
        lr = torch.nn.Linear(user_feat.shape[0], 1).to(device)
        user_feat_sig = torch.sigmoid(lr(user_feat)).to(device)
        user_feat_sig_1 = user_feat_sig.item() # [0, 1]

        r = beta.rvs(1, 1, size=1)[0]

        curr_delay = r * 10 + round(numpy.random.exponential(1))
        # curr_delay = r * 1
        curr_delay = round(curr_delay)

        curr_elap = random.randint(0, 10)

        curr_label = row["real_label"]
        # draw_delay_time.append(curr_delay)
        if curr_label == 1:
            real_delay_time.append(curr_delay)

        if row["real_label"] == 0:
            curr_delay = 0
        elif row["A"] == 1:
            if curr_elap <= curr_delay:
                curr_delay = 0
                curr_label = 0
        if curr_label == 1:
            sim_delay_time.append(curr_delay)
        elapsed_day.append(curr_elap)
        cv_delay_day.append(curr_delay)
        labels.append(curr_label)

    # print(count_good * 1.0 / total)
    real_delay_counter = Counter(real_delay_time)
    sim_delay_counter = Counter(sim_delay_time)
    real_draw_df = pd.DataFrame.from_dict(real_delay_counter, orient='index').sort_index()
    sim_delay_df = pd.DataFrame.from_dict(sim_delay_counter, orient='index').sort_index()
    draw_df = pd.concat([real_draw_df, sim_delay_df], ignore_index=True, sort=False)
    draw_df.plot(kind='bar')
    plt.savefig('Twins_train.pdf')


    df["label"] = labels
    df["elapsed_day"] = elapsed_day

    df["cv_delay_day"] = cv_delay_day
    df["supervised"] = labels

    df = df[df['elapsed_day'] > 3]
    df.to_csv("Twins_train.csv")

def produce_Twins_test():
    df = pd.read_csv("TWINS/TwinTest.csv")

    df["real_label"] = df["label"]
    elapsed_day = []
    cv_delay_day = []
    labels = []

    for index, row in df.iterrows():
        curr_elap = random.randint(0, 10)
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

    df["label"] = df["real_label"]
    df["elapsed_day"] = elapsed_day
    df["cv_delay_day"] = cv_delay_day
    df["supervised"] = labels
    df.to_csv("Twins_test.csv")
    
def verification_Twins_train():
    # orig_train = pd.read_csv("ACIC2019/epilepsyMod41.csv")
    new_train = pd.read_csv("Twins_train.csv")
    count_old = 0
    count_new = 0
    total = 0
    count_treated = 0
    count_control = 0
    for index, row in new_train.iterrows():
        total += 1
        if int(row["label"]) != int(row["real_label"]):
            count_new += 1
        if int(row["A"]) == 1:
            count_treated += 1
        else:
            count_control += 1
    print(count_treated)
    print(count_control)
    print(count_new * 1.0/total)

def verification_Twins_test():
    new_train = pd.read_csv("Twins_test.csv")
    count_old = 0
    count_new = 0
    total = 0
    count_treated = 0
    count_control = 0
    for index, row in new_train.iterrows():
        total += 1
        if int(row["label"]) != int(row["real_label"]):
            count_new += 1
        if int(row["A"]) == 1:
            count_treated += 1
        else:
            count_control += 1
    print(count_treated)
    print(count_control)
    print(count_new * 1.0/total)


if __name__ == '__main__':
    produce_Twins_train_beta()
    verification_Twins_train()
    # verification_Twins_test()
    # produce_Twins_test()
    # verification_Twins_test()
