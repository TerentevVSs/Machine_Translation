from translation_models import Attention_LSTM, Encoder_LSTM, \
    Decoder_LSTM, Translator_LSTM
from functions import *


device = torch.device('cpu')
input_dim = 11778
output_dim = 16483
encoder_embedding_dim = 256
decoder_embedding_dim = 256
encoder_hidden_dim = 512
decoder_hidden_dim = 512
encoder_dropout_prob = 0.5
decoder_dropout_prob = 0.5

attention = Attention_LSTM(encoder_hidden_dim, decoder_hidden_dim)
encoder = Encoder_LSTM(input_dim, encoder_embedding_dim,
                       encoder_hidden_dim,
                       decoder_hidden_dim, encoder_dropout_prob)
decoder = Decoder_LSTM(output_dim, decoder_embedding_dim,
                       encoder_hidden_dim,
                       decoder_hidden_dim, decoder_dropout_prob,
                       attention)

model = Translator_LSTM(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('attention_LSTM_bleu_en_ru.pt',
                                 map_location=device))


def translate_attention_LSTM(example, SRC, TRG, translator=model,
                          device=device):
    translation = translate(example, translator, TRG, SRC, device)
    return translation
