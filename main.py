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

def process():
    df = pd.read_csv("processed_test.csv")

    df['label'] = df['real_label']

    df.to_csv("processed_test_real_label.csv", index=False)


if __name__ == '__main__':
    #produce_ACIC2019_train()
    produce_ACIC2019_train_beta()
    verification_ACIC2019_train()





    # produce_ACIC2019_test()
    # processACIC2019()
    # verification_ACIC2019_test()
