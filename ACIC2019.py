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


def examine_ACIC2019():
    print("examine_ACIC2019")
    df = pd.read_csv("ACIC2019/epilepsyMod41.csv")
    count_y = 0
    count_row = 0
    count_a= 0
    for index, row in df.iterrows():
        count_row += 1
        if row["Y"] == 1:
            count_y += 1
        if row["A"] == 1:
            count_a += 1
    print("Train Count y percentage")
    print(count_y * 1.0 / count_row)

    print("Train Count a percentage")
    print(count_a * 1.0 / count_row)

def produce_ACIC2019_train_beta():
    df = pd.read_csv("ACIC2019/epilepsyMod41.csv")

    df["real_label"] = df["Y"]
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

        r = beta.rvs(0.5, 1, size=1)[0]

        curr_delay = r * 5 + round(numpy.random.exponential(0.5))
        # curr_delay = r * 1
        # curr_delay = r * 10
        curr_delay = round(curr_delay)

        curr_elap = random.randint(0, 10)

        curr_label = row["real_label"]
        # draw_delay_time.append(curr_delay)
        if row["A"] == 1 and curr_label == 1:
            real_delay_time.append(curr_delay)

        if row["real_label"] == 0:
            curr_delay = 0
        elif row["A"] == 1:
            if curr_elap <= curr_delay:
                curr_delay = 0
                curr_label = 0
        if row["A"] == 1 and curr_label == 1:
            sim_delay_time.append(curr_delay)
        elapsed_day.append(curr_elap)
        cv_delay_day.append(curr_delay)
        labels.append(curr_label)

    # print(count_good * 1.0 / total)
    real_delay_counter = Counter(real_delay_time)
    sim_delay_counter = Counter(sim_delay_time)
    real_draw_df = pd.DataFrame.from_dict(real_delay_counter, orient='index').sort_index()
    sim_delay_df = pd.DataFrame.from_dict(sim_delay_counter, orient='index').sort_index()
    # draw_df = pd.concat([real_draw_df, sim_delay_df], ignore_index=True, sort=False)
    # Plot the real_draw_df DataFrame
    plt.bar(real_draw_df.index, real_draw_df[0], color='blue', label='Real Delay')

    # Plot the sim_delay_df DataFrame
    plt.bar(sim_delay_df.index, sim_delay_df[0], color='orange', label='Observed Delay')

    # Customize the plot
    plt.xlabel('Delay Time')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Comparison of Real and Observed Delays')

    # Save the plot as a PDF file
    plt.savefig('ACIC2019_train.pdf')

    # Show the plot
    plt.show()


    df["label"] = labels
    df["elapsed_day"] = elapsed_day

    df["cv_delay_day"] = cv_delay_day
    df["supervised"] = labels

    # df = df[df['elapsed_day'] > 2]
    df.to_csv("ACIC2019_train.csv")

def produce_ACIC2019_test():
    df = pd.read_csv("ACIC2019/epilepsyMod42.csv")

    df["real_label"] = df["Y"]
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

    df["label"] = labels
    df["elapsed_day"] = elapsed_day
    df["cv_delay_day"] = cv_delay_day
    df["supervised"] = labels
    df.to_csv("ACIC2019_test.csv")
    
def verification_ACIC2019_train():
    # orig_train = pd.read_csv("ACIC2019/epilepsyMod41.csv")
    new_train = pd.read_csv("ACIC2019_train.csv")
    count_old = 0
    count_new = 0
    total = 0
    for index, row in new_train.iterrows():
        total += 1
        if int(row["A"] == 1):
            count_old += 1
        if int(row["label"]) != int(row["real_label"]):
            count_new += 1
    print(count_new * 1.0/total)
    print(count_new * 1.0/count_old)

def verification_ACIC2019_test():
    orig_test = pd.read_csv("ACIC2019/epilepsyMod42.csv")
    new_test = pd.read_csv("ACIC2019_test_real_label.csv")
    count_old = 0
    count_new = 0
    total = 0
    for index, row in orig_test.iterrows():
        total += 1
        if int(row["Y"]) != int(new_test.iloc[[index]]["label"]):
            print(row["Y"])
            print(new_test.iloc[[index]]["label"])
            count_new += 1
    print(count_new * 1.0/total)



if __name__ == '__main__':
    #produce_ACIC2019_train()
    produce_ACIC2019_train_beta()
    verification_ACIC2019_train()
