import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
import random


class Translator(nn.Module):
    def __init__(self,
                 encoder_obj,
                 decoder_obj,
                 from_vocab,
                 to_vocab,
                 teacher_force_ratio,
                 device):

        super(Translator, self).__init__()
        self.teacher_force_ratio = teacher_force_ratio
        self.device = device
        self.to_vocab = to_vocab
        self.from_vocab = from_vocab
        self.encoder = encoder_obj

        self.decoder = decoder_obj

    def forward(self, source, target):
        english = self.to_vocab
        german = self.from_vocab
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english)

        outputs = torch.zeros(target_len, batch_size,
                              target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            # print("OK 43")
            output, hidden, cell = self.decoder(x, hidden, cell)
            # print("Decoder ok")
            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            x = target[t] if random.random(
            ) < self.teacher_force_ratio else best_guess

        return outputs
