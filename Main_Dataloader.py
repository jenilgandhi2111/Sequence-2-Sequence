from os import name
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.legacy import data
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy
from Dataloader import get_loader
from Translator import Translator
from Decoder import Decoder
from Encoder import Encoder
from tqdm import tqdm
from utils import translate_sentence


'''
For the file "Main_Dataloader.py" I have used the custom dataloader class
built using torchtext and spacy.
'''

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("> Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    loader, dataset = get_loader("english.tsv", "german.tsv", batch_size=64)
    # HyperParams
    EPOCHS = 2
    lr = 1e-3
    batch_size = 64
    enc_input_size = len(dataset.ger_vocab)
    dec_input_size = len(dataset.eng_vocab)
    dec_output_size = len(dataset.eng_vocab)
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

    # Encoder and decoder objects to be passed to the Translator
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
        from_vocab=dataset.ger_vocab,
        to_vocab=dataset.eng_vocab,
        teacher_force_ratio=teacher_force_ratio,
        device=device)

    # Optimizer and loss function
    optimizer = optim.Adam(translator.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=dataset.eng_vocab.stoi["<PAD>"])

    # Test Sentence
    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

    print("> Starting Training")
    for epoch in range(EPOCHS):

        print(f'Epoch:[{epoch+1}/{EPOCHS}]')
        translator.train()

        for batch_id, (y, x) in tqdm(enumerate(loader), total=len(loader), leave=False):
            input_data = x.to(device)
            targets = y.to(device)
            output = translator(input_data, targets)
            output = output[1:].reshape(-1, output.shape[2])
            targets = targets[1:].reshape(-1)
            optimizer.zero_grad()
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

        # translator.eval()
        # print(translate_sentence(translator,
        #                          sentence,
        #                          dataset.ger_vocab,
        #                          dataset.eng_vocab,
        #                          device,
        #                          50))
