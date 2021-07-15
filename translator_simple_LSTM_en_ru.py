from translation_models import Encoder_simple, Decoder_simple, \
    Translator_simple
from functions import *


device = torch.device('cpu')
input_dim = 11778
output_dim = 16483
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
layers = 2
encoder_dropout_prob = 0.5
decoder_dropout_prob = 0.5
bidirectional = True

encoder = Encoder_simple(input_dim, encoder_embedding_dim,
                         hidden_dim // 2, layers,
                         encoder_dropout_prob,
                         bidirectional=bidirectional)
decoder = Decoder_simple(output_dim, decoder_embedding_dim,
                         hidden_dim, layers, decoder_dropout_prob)

model = Translator_simple(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('simple_LSTM_bleu_en_ru.pt',
                                 map_location=device))


def translate_simple_LSTM(example, SRC, TRG, translator=model,
                          device=device):
    translation = translate(example, translator, SRC, TRG, device)
    return translation
