
""" 
Data management and loaders
"""

import os
import string
import numpy as np 
import pandas as pd
from ntpath import basename
from transformers import BertTokenizer


class DataLoader:

    def __init__(self, titles, abstracts, batch_size, shuffle, device):
        self.ptr = 0
        self.titles = titles
        self.device = device
        self.abstracts = abstracts
        self.batch_size = batch_size 
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if shuffle:
            idx = np.random.permutation(np.arange(len(self.titles)))
            self.titles = np.asarray(self.titles)[idx].tolist()
            self.abstracts = np.asarray(self.abstracts)[idx].tolist()

    def __len__(self):
        return len(self.titles) // self.batch_size

    def flow(self):
        titles, abstracts = [], []
        for _ in range(self.batch_size):
            titles.append(self.titles[self.ptr])
            abstract = self.abstracts[self.ptr]
            abstracts.append(abstract.translate(str.maketrans({key: " {0}".format(key) for key in string.punctuation})))
            self.ptr += 1

            if self.ptr >= len(self.titles):
                self.ptr = 0

        title_tokens = self.tokenizer(titles, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(self.device)
        if isinstance(abstracts[0], str):
            abstract_tokens = self.tokenizer(abstracts, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(self.device)
            return titles, title_tokens, abstracts, abstract_tokens
        else:
            return titles, title_tokens, abstracts


def get_dataloaders(root, val_split, batch_size, device):
    """ 
    root will have train.csv and test.csv
    """
    main = pd.read_csv(os.path.join(root, "train.csv"))
    test = pd.read_csv(os.path.join(root, "test.csv"))

    main_titles, main_abstracts = main['title'].values.tolist(), main['abstract'].values.tolist()
    test_titles = test['title'].values.tolist()
    assert len(main_titles) == len(main_abstracts), f"Train data has {len(train_titles)} titles and {len(train_abstracts)} abstracts"

    val_size = int(val_split * len(main_titles))
    val_titles, val_abstracts = main_titles[:val_size], main_abstracts[:val_size]
    train_titles, train_abstracts = main_titles[val_size:], main_abstracts[val_size:]

    train_loader = DataLoader(train_titles, train_abstracts, batch_size=batch_size, shuffle=True, device=device)
    val_loader = DataLoader(val_titles, val_abstracts, batch_size=batch_size, shuffle=False, device=device)
    test_loader = DataLoader(test_titles, np.arange(len(test_titles)), batch_size=batch_size, shuffle=False, device=device)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    root = "../track2_data"
    train = pd.read_csv(os.path.join(root, "train.csv"))
    test = pd.read_csv(os.path.join(root, "test.csv"))

    print(train.columns)
    print(test.columns)