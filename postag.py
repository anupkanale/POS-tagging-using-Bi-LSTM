import torch
import torch.nn as nn
import json
from utils import prep_sequence
import sys
from poslearn import lstm_posTagger


def read_test_data(test_path):
    x_tokens = []
    sentence_lengths = []

    f = open(test_path, 'r', encoding='utf8')
    all_lines = f.readlines()
    f.close()
    for line in all_lines:
        tokens = line.split()
        sentence_lengths.append(len(tokens))
        x_tokens.append(tokens)

    return x_tokens, sentence_lengths


def write_tagged_file(output_file_path, x, y, lengths, idx_to_tags):
    f = open(output_file_path, 'w')
    for ii in range(len(x)):
        tokens = []
        for jj in range(lengths[ii]):
            token = x[ii][jj]
            label = idx_to_tags[str(y[ii][jj])].upper()
            tokens.append(token + "/" + label)
        sentence = " ".join(tokens)
        f.write(sentence)
        f.write('\n')
    f.close()


if __name__=="__main__":
    MAX_LENGTH = 100

    test_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Load dictionaries
    with open('idx_to_tokens.json', 'r') as fp:
        idx_to_tokens = json.load(fp)
    with open('idx_to_tags.json', 'r') as fp:
        idx_to_tags = json.load(fp)
    with open('tokens_to_idx.json', 'r') as fp:
        tokens_to_idx = json.load(fp)
    with open('tags_to_idx.json', 'r') as fp:
        tags_to_idx = json.load(fp)

    # Load trained model
    model = torch.load('pos_tagging_model.pt')

    # Read test data
    x_test, orig_lengths = read_test_data(test_path)
    # print(x_test)
    # print(orig_lengths)

    # Generate padded sequences from test data
    x_test_padded = prep_sequence(x_test, orig_lengths, len(x_test), MAX_LENGTH, tokens_to_idx)
    test_lengths_tensor = torch.LongTensor(orig_lengths).to(torch.int64)
    # print("Padded x:", x_test_padded)
    # print("Padded lengths:", test_lengths_tensor)

    # PERFORMANCE ON TEST DATA
    with torch.no_grad():
        # Calculate performance on test data
        dev_prediction_tensor = model(x_test_padded, test_lengths_tensor, MAX_LENGTH)
        y_pred = torch.argmax(dev_prediction_tensor, dim=2)
        y_pred = y_pred.tolist()

    # ind = 1
    # print(y_pred[ind][0:orig_lengths[ind]])
    # print(x_test[ind])
    # print(idx_to_tags['2'])

    # Write to tagged output file
    write_tagged_file(output_file_path, x_test, y_pred, orig_lengths, idx_to_tags)

