from translation_models import Attention_GRU, Encoder_GRU, Decoder_GRU, \
    Translator_GRU
from functions import *


device = torch.device('cpu')
input_dim = 16483
output_dim = 11778
encoder_embedding_dim = 256
decoder_embedding_dim = 256
encoder_hidden_dim = 512
decoder_hidden_dim = 512
encoder_dropout_prob = 0.5
decoder_dropout_prob = 0.5

attention = Attention_GRU(encoder_hidden_dim, decoder_hidden_dim)
encoder = Encoder_GRU(input_dim, encoder_embedding_dim,
                      encoder_hidden_dim,
                      decoder_hidden_dim, encoder_dropout_prob)
decoder = Decoder_GRU(output_dim, decoder_embedding_dim,
                      encoder_hidden_dim,
                      decoder_hidden_dim, decoder_dropout_prob,
                      attention)

model = Translator_GRU(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('attention_GRU_bleu_ru_en.pt',
                                 map_location=device))


def translate_attention_GRU(example, SRC, TRG, translator=model,
                          device=device):
    translation = translate(example, translator, TRG, SRC, device)
    return translation
