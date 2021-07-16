import torch
import torch.nn as nn
import numpy as np
from torchtext.legacy.data import Field, BucketIterator
import random
from nltk.tokenize import WordPunctTokenizer
import torchtext
from torch.nn import functional as F
import os


def _len_sort_key(x):
    return len(x.src)


def temp_softmax(x, dim=0, temperature=1):
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=dim)


def tokenize_ru(x, tokenizer=WordPunctTokenizer()):
    return tokenizer.tokenize(x.lower())


def tokenize_en(x, tokenizer=WordPunctTokenizer()):
    return tokenizer.tokenize(x.lower())


def delete_eos(tokens_iter):
    for token in tokens_iter:
        if token == '<eos>':
            break
        yield token


def remove_tech_tokens(tokens_iter,
                       tokens_to_remove=['<sos>', '<unk>', '<pad>']):
    return [x for x in tokens_iter if x not in tokens_to_remove]


def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    # запускаем без teacher_forcing
    output = model(src, trg, 0)
    # удаляем первый токен и выбираем лучшее слово
    output = output[1:].argmax(-1)
    data = [TRG_vocab.itos[x] for x in list(output[:, 0].cpu().numpy())]
    generated = remove_tech_tokens(delete_eos(data))
    return 'Перевод модели: {}'.format(' '.join(generated)), \
           'Перевод модели с учетом неизвестных слов: {}'.format(
               ' '.join(data))


def get_text(x, TRG_vocab):
    generated = remove_tech_tokens(
        delete_eos([TRG_vocab.itos[elem] for elem in list(x)]))
    return generated


def translate(example, translator, TRG, SRC, device):
    with open('example.csv', 'w', encoding='utf-8') as file:
        example = str(example)
        example = str(example.replace('\n', '').replace('\r', ''))
        examples = str(example + ' ' + example + ',' + example)
        file.write(examples)
    test_dataset = torchtext.legacy.data.TabularDataset(
        path='example.csv',
        format='csv',
        fields=[('trg', TRG), ('src', SRC)]
    )

    iterator = BucketIterator(
        test_dataset,
        batch_size=1,
        device=device,
        sort_key=_len_sort_key
    )
    generated_text = []
    translator.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            # запускаем без teacher_forcing
            output = translator(src, trg, 0)
            # удаляем первый токен и выбираем лучшее слово
            output = output[1:].argmax(-1)
            generated_text.extend([get_text(x, TRG.vocab) for x in
                                   output.detach().cpu().numpy().T])
            generated_text = (' '.join(generated_text[0])[
                              :-2] + '.').capitalize()
    translation = 'Перевод модели: {}'.format(generated_text)
    os.remove('example.csv')
    return translation


def get_data():
    SEED = 666

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cpu')

    SRC = Field(tokenize=tokenize_ru,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    dataset = torchtext.legacy.data.TabularDataset(
        path='data.txt',
        format='tsv',
        fields=[('trg', TRG), ('src', SRC)]
    )

    SRC.build_vocab(dataset, min_freq=2)
    TRG.build_vocab(dataset, min_freq=2)
    return SRC, TRG
