from functions import *


# Attention encoder-decoder with LSTM
class Encoder_LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                 dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.LSTM = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True,
                            num_layers=2)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.fc_2 = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.LSTM(embedded)
        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        hidden = torch.tanh(self.fc(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        cell = torch.tanh(self.fc_2(
            torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))
        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        # cell = [n layers, batch size, hid dim]
        return outputs, hidden, cell


class Attention_LSTM(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim,
                              dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)


    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1)
        # hidden = [batch size, 1, dec hid dim]

        hidden = hidden.repeat(1, src_len, 1)
        # hidden = [batch size, src len, dec hid dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]

        return temp_softmax(attention, dim=1)


class Decoder_LSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                 dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.LSTM = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        hidden = hidden.unsqueeze(0)
        # hidden = [1, batch size, dec hid dim]
        cell = cell.unsqueeze(0)
        # cell = [1, batch size, dec hid dim]

        output, (hidden, cell) = self.LSTM(rnn_input, (hidden, cell))
        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]
        # cell = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will be 1
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # cell = [1, batch size, dec hid dim]

        embedded = embedded.squeeze(0)
        # embedded = [batch size, emb dim]
        output = output.squeeze(0)
        # output = [batch size, dec hid dim]
        weighted = weighted.squeeze(0)
        # weighted = [batch size, enc hid dim * 2]
        hidden = hidden.squeeze(0)
        # hidden = [batch size, dec hid dim]
        cell = cell.squeeze(0)
        # cell = [batch size, dec hid dim]

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Translator_LSTM(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing):
        # src = [src len, batch size]
        # trg = [trg len, batch size]

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(
            self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, cell = self.decoder(input, hidden, cell,
                                                encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

# Attention encoder-decoder with GRU
class Encoder_GRU(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                 dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        hidden = torch.tanh(self.fc(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention_GRU(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim,
                              dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch size, src len, dec hid dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder_GRU(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                 dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)


class Translator_GRU(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing):
        # src = [src len, batch size]
        # trg = [trg len, batch size]

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(
            self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden,
                                          encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


# Simple encoder-decoder with LSTM
class Encoder_simple(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers,
                 dropout_prob, bidirectional=False):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                           dropout=dropout_prob,
                           bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded)
        # output = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        if self.bidirectional:
            hidden = hidden.reshape(self.n_layers, 2, -1, self.hid_dim)
            hidden = hidden.transpose(1, 2).reshape(self.n_layers, -1,
                                                    2 * self.hid_dim)

            cell = cell.reshape(self.n_layers, 2, -1, self.hid_dim)
            cell = cell.transpose(1, 2).reshape(self.n_layers, -1,
                                                2 * self.hid_dim)

        return hidden, cell


class Decoder_simple(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers,
                 dropout_prob):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                           dropout=dropout_prob)

        self.out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        unsqueeze_input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(unsqueeze_input))
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [1, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.out(output.squeeze(0))

        return prediction, hidden, cell


class Translator_simple(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing):
        # src = [src len, batch size]
        # trg = [trg len, batch size]

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(
            self.device)

        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for token in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[token] = output
            teacher_force = random.random() < teacher_forcing
            top = output.argmax(-1)
            input = trg[token] if teacher_force else top
        return outputs

