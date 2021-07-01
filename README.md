# Sequence-2-Sequence
## Running Example: 
German to English Translation
## Description: 
Sequence to sequence is a model based on encoder decoder architecture, Where the Encoder 
encoder encodes the german sentence by passing it to the lstm cells which in the last cell generates the features via the hidden states of the encoder. Now the decoder is then fed this hidden state as initial hidden state and a \<SOS\> token now after each lstm cell it is mapped to a hidden size vector , now this hidden size vector is then mapped to vocab size vector using a dense layer (in code used as fc_out in file_decoder.py).
## Dataset used:
The dataset used in this code is the Multi30K provided by the torchtext.legacy.datasets, which used to be a benchmark for machine translation. This dataset contains 29000 english and german translation sentences for testing , 1000 for testing and 1010 for validation.
## How to use?
<ul>
<li>I made the custom dataloader class for the dataloading process and implementing it in Main_Dataloader.py</li>
<li>For the easy implementation Run the Main.py</li>
</ul>
