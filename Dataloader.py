import os
import sys
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, dataset

spacy_ger = spacy.load("de_core_news_md")
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self,
                 language,
                 freq_threshhold):
        self.language = language
        if language == "de":
            self.language_model = spacy_ger
        else:
            self.language_model = spacy_eng

        self.freq_threshhold = freq_threshhold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenizer(sent, tokenizer_fn):

        # print([tok.text for tok in tokenizer_fn.tokenizer(sent[0])])
        return [tok.text for tok in tokenizer_fn.tokenizer(sent[0])]

    def build_vocab(self, sentence_list):
        print("> Building vocab for:", self.language)

        freq = {}
        idx = 4
        cntr = 0
        for i in sentence_list:
            # print(cntr)
            cntr += 1
            for word in self.tokenizer(i, self.language_model):
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1

                if freq[word] == 2:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                    # print(idx)
        print("> Finised Building vocab for:", self.language)
        # print(self.stoi)

    def numericalize(self, text):

        # print(self.stoi[text.split(" ")[0]])
        # print(text.split(" ")[0])
        # print(self.stoi[text.split(" ")[1]])
        # print(text.split(" ")[1])
        # print(text)
        # print(self.stoi)
        # print(self.itos[24])
        # print(self.stoi['ballet'])
        # print(self.itos[3755])
        ret_lst = []
        for word in self.language_model.tokenizer(text):
            # print(word)
            if str(word) in self.stoi:
                ret_lst.append(self.stoi[str(word)])
            else:
                ret_lst.append(self.stoi["<UNK>"])
        # print(ret_lst)
        return ret_lst


class MyDataset(Dataset):
    def __init__(self, eng_file, ger_file, freq_threshhold=2):
        self.eng_file = eng_file
        self.ger_file = ger_file
        self.english = (pd.read_csv(
            eng_file, delimiter="\n")).values.tolist()
        self.german = (pd.read_csv(
            ger_file, delimiter="\n")).values.tolist()

        self.eng_vocab = Vocabulary("en", 2)
        self.ger_vocab = Vocabulary("de", 2)
        self.eng_vocab.build_vocab(self.english)
        self.ger_vocab.build_vocab(self.german)

    def __len__(self):
        return len(self.english)

    def __getitem__(self, index):
        # print("index:", index)
        eng_sent = self.english[index][0]
        # print(eng_sent)
        num_eng_sent = [self.eng_vocab.stoi["<SOS>"]]
        num_eng_sent += self.eng_vocab.numericalize(eng_sent)
        num_eng_sent.append(self.eng_vocab.stoi["<EOS>"])
        # print(len(num_eng_sent))

        ger_sent = self.german[index][0]
        num_ger_sent = [self.ger_vocab.stoi["<SOS>"]]
        num_ger_sent += self.ger_vocab.numericalize(ger_sent)
        num_ger_sent.append(self.ger_vocab.stoi["<EOS>"])
        # print(torch.tensor(num_eng_sent).shape)
        return torch.tensor(num_eng_sent), torch.tensor(num_ger_sent)


class MyCollate:
    def __init__(self, pad_id_eng, pad_id_ger):
        self.pad_id_eng = pad_id_eng
        self.pad_id_ger = pad_id_ger

    def __call__(self, batch):
        srcs = [item[0] for item in batch]
        srcs = pad_sequence(srcs, batch_first=False,
                            padding_value=self.pad_id_ger)
        trgs = [item[1] for item in batch]
        trgs = pad_sequence(trgs, batch_first=False,
                            padding_value=self.pad_id_eng)
        return srcs, trgs


def get_loader(english_file,
               german_file,
               batch_size=32,
               num_workers=2,
               shuffle=False,
               pin_memory=True):

    dataset = MyDataset(english_file, german_file, 2)
    pad_id_eng = dataset.eng_vocab.stoi["<PAD>"]
    pad_id_ger = dataset.ger_vocab.stoi["<PAD>"]

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        pin_memory=pin_memory,
                        collate_fn=MyCollate(pad_id_eng=pad_id_eng, pad_id_ger=pad_id_ger))

    return loader, dataset


# if __name__ == "__main__":
#     loader, dataset = get_loader("english.tsv",
#                                  "german.tsv")
#     cntr = 0
#     for idx, (a, b) in enumerate(loader):
#         if cntr == 0:
#             cntr += 1
#             print(a.shape)
#             print(b.shape)
#             print(a)

    # print(a.shape)
# md = MyDataset("english.tsv", "german.tsv", 2)
# print(md[9])
