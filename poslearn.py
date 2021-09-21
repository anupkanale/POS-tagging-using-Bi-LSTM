import torch
import torch.nn as nn
import sklearn.metrics as metrics
import torch.optim as optim
from utils import read_data, prep_sequence
import json
import sys
# import time
# import numpy as np


class lstm_posTagger(nn.Module):
    def __init__(self, N_TOKENS, N_TAGS, EMBED_DIM, HIDDEN_DIM):
        super(lstm_posTagger, self).__init__()

        self.embedding = nn.Embedding(N_TOKENS, EMBED_DIM)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=1, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(HIDDEN_DIM*2, N_TAGS)

    def forward(self, sentences, lens, max_length):
        # print("Input size BEFORE embedding: ", torch.Tensor.size(sentences))
        embeds = self.embedding(sentences)
        # print("Input size AFTER embedding: ", torch.Tensor.size(embeds))

        packed = nn.utils.rnn.pack_padded_sequence(embeds, lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=max_length)
        # print("After padding packed seq: ", torch.Tensor.size(lstm_out))

        predicted_tags = self.hidden2tag(lstm_out)
        # print("After linear layer: ", torch.Tensor.size(predicted_tags))
        return predicted_tags


if __name__=="__main__":
    # start_time = time.time()
    # torch.manual_seed(0)  # for repeatability

    # ------------------------------------
    # DEFINE HYPER-PARAMETERS
    # ------------------------------------
    NUM_EPOCHS = 20
    EMBED_DIM = 100
    HIDDEN_DIM = 100
    MAX_LENGTH = 100
    BATCH_SIZE = 100
    LEARNING_RATE = 0.001

    train_path = sys.argv[1]
    dev_path = sys.argv[2]

    # ------------------------------------
    # PRE-PROCESS: TRAINING DATA
    # ------------------------------------
    # Read training data
    x_train, y_train, train_lengths, tokens_to_idx, tags_to_idx, idx_to_tokens, idx_to_tags =\
        read_data(train_path, True)
    N_TOKENS = len(tokens_to_idx)
    N_TAGS = len(tags_to_idx)
    N_EXAMPLES = len(train_lengths)
    # print("Training features: ", x_train)
    # print("Training tags: ", y_train)

    # Pad sequences from traning data
    x_train_padded = prep_sequence(x_train, train_lengths, N_EXAMPLES, MAX_LENGTH, tokens_to_idx)
    y_train_padded = prep_sequence(y_train, train_lengths, N_EXAMPLES, MAX_LENGTH, tags_to_idx)
    train_lengths_tensor = torch.LongTensor(train_lengths).to(torch.int64)
    # print("Training features: ", x_padded[0])
    # print("Training tags: ", y_padded[0])

    # ------------------------------------
    # PRE-PROCESS: DEV DATA
    # ------------------------------------
    # Read dev data
    x_dev, y_dev, dev_lengths, _, _, _, _ = read_data(dev_path, False)
    N_DEV = len(y_dev)
    # print("Dev features: ", x_dev)
    # print("Dev tags: ", y_dev)

    # Generate padded sequences from dev data
    x_dev_padded = prep_sequence(x_dev, dev_lengths, N_DEV, MAX_LENGTH, tokens_to_idx)
    y_dev_padded = prep_sequence(y_dev, dev_lengths, N_DEV, MAX_LENGTH, tags_to_idx)
    dev_lengths_tensor = torch.LongTensor(dev_lengths).to(torch.int64)

    y_dev_padded = y_dev_padded.view(N_DEV * MAX_LENGTH, )
    y_dev_padded = y_dev_padded.tolist()
    # print(x_dev_padded)
    # print(y_dev_padded)
    # print(torch.Tensor.size(x_dev_padded))

    # ------------------------------------
    # ML MODEL-- Bi-LSTM
    # ------------------------------------
    # Build model
    model = lstm_posTagger(N_TOKENS, N_TAGS, EMBED_DIM, HIDDEN_DIM)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # print(model.parameters)
    # print(loss_func)
    # print(optimizer)

    # Train model
    for ii in range(NUM_EPOCHS):
        for jj in range(N_EXAMPLES//BATCH_SIZE):
            batch_indices = torch.randperm(N_EXAMPLES)[:BATCH_SIZE]
            x_batch = x_train_padded[batch_indices, :]
            y_batch = y_train_padded[batch_indices, :]
            # print(x_batch)
            # print(y_batch)
            # print(lens[batch_indices])

            model.zero_grad()

            predictions = model(x_batch, train_lengths_tensor[batch_indices], MAX_LENGTH)
            predicted_tags = torch.argmax(predictions, dim=2)
            # print("Predicted tags: ", predicted_tags[0:5, 0:6])
            # print("True tags: ", y_batch[0:5, 0:6])

            predictions = predictions.view(BATCH_SIZE*MAX_LENGTH, N_TAGS)
            y_batch = y_batch.view(BATCH_SIZE*MAX_LENGTH, )

            train_loss = loss_func(predictions, y_batch)
            train_loss.backward()
            optimizer.step()

        # Test performance on dev data
        with torch.no_grad():
            # Calculate performance on Development data
            dev_prediction_tensor = model(x_dev_padded, dev_lengths_tensor, MAX_LENGTH)
            dev_prediction_tags = torch.argmax(dev_prediction_tensor, dim=2)
            dev_prediction_tags = dev_prediction_tags.view(N_DEV * MAX_LENGTH, )
            dev_prediction_tags = dev_prediction_tags.tolist()
            # print(dev_prediction_tags)
            # print(y_dev_padded[0:3], y_dev_padded[100:103],  y_dev_padded[100:103])

            y_dev_new = []
            dev_prediction_new = []
            for jj in range(len(y_dev_padded)):
                if y_dev_padded[jj] != 0 and y_dev_padded[jj] != 1:
                    y_dev_new.append(y_dev_padded[jj])
                    dev_prediction_new.append(dev_prediction_tags[jj])
            # print(y_dev_new)
            # print(dev_prediction_new)

            dev_F1 = metrics.f1_score(y_dev_new, dev_prediction_new, average='macro')
            accu_F1 = metrics.accuracy_score(y_dev_new, dev_prediction_new, normalize=True)
            print("epoch #%d:, training loss= %f, F1-score= %f, Accuracy= %f\n" % (ii + 1, train_loss.item(), dev_F1, accu_F1))

    # print("\nTotal time: ", time.time() - start_time)
    torch.save(model, 'pos_tagging_model.pt')
    with open('idx_to_tokens.json', 'w') as fp:
        json.dump(idx_to_tokens, fp)
    with open('idx_to_tags.json', 'w') as fp:
        json.dump(idx_to_tags, fp)
    with open('tokens_to_idx.json', 'w') as fp:
        json.dump(tokens_to_idx, fp)
    with open('tags_to_idx.json', 'w') as fp:
        json.dump(tags_to_idx, fp)

    print("Finished traning and saving files!")
