import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy
from Translator import Translator
from Decoder import Decoder
from Encoder import Encoder
from tqdm import tqdm
from utils import translate_sentence


'''
For Dataloading purpose we could also use our custom 'get_loader' function for preprocessing
But for now we would use the inbuilt Multi30K
'''

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Using device:", device)
    spacy_ger = spacy.load("de_core_news_md")
    spacy_eng = spacy.load("en_core_web_sm")
    # Defining the tokernizers for languages

    def tokenize_ger(text):
        '''
        Tokenizes German sentences and returns a list of tokenized sentences
        '''
        return [tok.text for tok in spacy_ger.tokenizer(text)]

    def tokenize_eng(text):
        '''
        Tokenizes English sentences and returns a list of tokenized sentences
        '''
        return [tok.text for tok in spacy_eng.tokenizer(text)]

    # Defining the vocab for languages the Field have utils for language
    print("> Building Field for german")
    german = Field(tokenize=tokenize_ger, tokenizer_language="de", lower=True,
                   init_token="<sos>", eos_token="<eos>")

    print("> Building Field for English")
    english = Field(
        tokenize=tokenize_eng, lower=True, init_token="<sos>", tokenizer_language="en", eos_token="<eos>"
    )

    # Dividing the data in train validation and testing
    print("> Splitting Data")
    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(german, english)
    )

    # Building a vocab
    print("> Building Vocab")
    german.build_vocab(train_data, max_size=10000, min_freq=2)
    english.build_vocab(train_data, max_size=10000, min_freq=2)
    print("> Finished bulding vocab")

    # Getting iterators
    print("> Building the iterators")
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=64,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )

    print("> Finished building iterators")

    # HyperParams
    EPOCHS = 2
    lr = 1e-3
    batch_size = 64
    enc_input_size = len(german.vocab)
    dec_input_size = len(english.vocab)
    dec_output_size = len(english.vocab)
    enc_embed_size = 256
    dec_embed_size = 256
    enc_dropout = 0.3
    dec_dropout = 0.3
    # Needs to be same for both encoder and decoder
    enc_hidden_size = 512
    dec_hidden_size = 512
    # Num layers for decoder and encoder must be same
    enc_num_layers = 2
    dec_num_layers = 2
    # Teacher forcing
    teacher_force_ratio = 0.5
    # Optimizer

    encoder = Encoder(input_size=enc_input_size,
                      embed_size=enc_embed_size,
                      hidden_size=enc_hidden_size,
                      num_layers=enc_num_layers,
                      dropout_p=enc_dropout)
    decoder = Decoder(input_size=dec_input_size,
                      embed_size=dec_embed_size,
                      hidden_size=dec_hidden_size,
                      num_layers=dec_num_layers,
                      output_size=dec_output_size,
                      dropout_p=dec_dropout)

    # Defining the model
    translator = Translator(
        encoder_obj=encoder,
        decoder_obj=decoder,
        from_vocab=german.vocab,
        to_vocab=english.vocab,
        teacher_force_ratio=teacher_force_ratio,
        device=device)
    optimizer = optim.Adam(translator.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=english.vocab.stoi["<PAD>"])
    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

    print("> Starting Training")
    for epoch in range(EPOCHS):
        print(f'Epoch:[{epoch+1}/{EPOCHS}]')

        translator.train()
        for batch_id, batch in tqdm(enumerate(train_iterator), total=len(train_iterator), leave=False):
            input_data = batch.src.to(device)
            targets = batch.trg.to(device)
            output = translator(input_data, targets)
            output = output[1:].reshape(-1, output.shape[2])
            targets = targets[1:].reshape(-1)
            optimizer.zero_grad()
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
        # translator.eval()
        # print(translate_sentence(translator, sentence, german, english, device, 50))
