
""" 
Data management and loaders
"""

import os
import numpy as np 
import pandas as pd
from ntpath import basename
from transformers import BartTokenizer


class DataLoader:

    def __init__(self, titles, abstracts, batch_size, shuffle, device):
        self.titles = titles
        self.device = device
        self.abstracts = abstracts
        self.batch_size = batch_size 
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.ptr = 0

        if shuffle:
            idx = np.random.permutation(np.arange(len(self.titles)))
            self.titles = self.titles[idx]
            self.abstracts = self.abstracts[idx]

    def __len__(self):
        return len(self.titles) // self.batch_size

    def flow(self):
        titles, abstracts = [], []
        for _ in range(self.batch_size):
            titles.append(self.titles[self.ptr])
            abstracts.append(self.abstracts[self.ptr])
            self.ptr += 1

            if self.ptr >= len(self.titles):
                self.ptr = 0

        title_tokens = self.tokenizer(titles, padding=True, return_tensors="pt")["input_ids"].to(self.device)
        if isinstance(abstracts[0], str):
            abstract_tokens = self.tokenizer(abstracts, padding=True, return_tensors="pt")["input_ids"].to(self.device)
            return titles, title_tokens, abstracts, abstract_tokens
        else:
            return titles, title_tokens, abstracts


def get_dataloaders(root, val_split, batch_size):
    """ 
    root will have train.csv and test.csv
    """
    main = pd.read_csv(os.path.join(root, "train.csv"))
    test = pd.read_csv(os.path.join(root, "test.csv"))

    main_titles, main_abstracts = main['title'].values.tolist(), main['abstract'].values.tolist()
    test_titles = test['title'].values.tolist()
    assert len(main_titles) == len(main_abstracts), f"Train data has {len(train_titles)} titles and {len(train_abstracts)} abstracts"

    val_size = int(val_split * len(main_titles))
    val_titles, val_abstracts = main_titles[:val_split], main_abstracts[:val_split]
    train_titles, train_abstracts = main_titles[val_split:], main_abstracts[val_split:]

    train_loader = DataLoader(train_titles, train_abstracts, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_titles, val_abstracts, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_titles, np.arange(len(test_titles)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    root = "../track2_data"
    train = pd.read_csv(os.path.join(root, "train.csv"))
    test = pd.read_csv(os.path.join(root, "test.csv"))

    print(train.columns)
    print(test.columns)