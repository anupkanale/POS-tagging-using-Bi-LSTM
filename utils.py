import torch
# import numpy as np


def read_data(path_to_file, training):
    tokens_to_idx = {"pad": 0, "unk": 1}  # key:vocabulary, value:index
    tags_to_idx = {"pad": 0, "unk": 1}  # key:pos, value:index
    idx_to_tokens = {0: "pad", 1: "unk"}
    idx_to_tags = {0: "pad", 1: "unk"}
    x_tokens = []  # list of tuples: (sentence, corresponding tags, length of sentence)
    y_tags = []
    sentence_lengths = []

    f = open(path_to_file, 'r', encoding='utf8')
    all_lines = f.readlines()
    f.close()
    for line in all_lines:
        sentence_tokens = []
        sentence_tags = []
        for pair in line.split():
            word = pair.split('/')[0].lower()
            sentence_tokens.append(word)

            tag = pair.split('/')[1].lower()
            sentence_tags.append(tag)

            if training:
                # Add to vocab if not seen before
                if word not in tokens_to_idx:
                    idx_to_tokens[len(tokens_to_idx)] = word
                    tokens_to_idx[word] = len(tokens_to_idx)

                # Add to list of unique tags if not seen before
                if tag not in tags_to_idx:
                    idx_to_tags[len(tags_to_idx)] = tag
                    tags_to_idx[tag] = len(tags_to_idx)

        x_tokens.append(sentence_tokens)
        y_tags.append(sentence_tags)
        sentence_lengths.append(len(sentence_tokens))

    return x_tokens, y_tags, sentence_lengths, tokens_to_idx, tags_to_idx, idx_to_tokens, idx_to_tags


def token_count(words_to_idx):
    token_freq = {}
    for key in words_to_idx:
        token_freq[key] = 0

    f = open('train.txt', 'r')
    all_lines = f.readlines()
    for line in all_lines:
        for pair in line.split():
            token = pair.split('/')[0].lower()
            token_freq[token] += 1

    print("Total number of tokens", len(token_freq))


def to_vector(seq, map_to_idx):
    vec = []
    for token in seq:
        if token in map_to_idx:
            vec.append(map_to_idx[token])
        else:
            vec.append(map_to_idx["unk"])
    return torch.tensor(vec)


def prep_sequence(x, orig_lengths, num_sequences, max_length, tokens_to_idx):
    # Generate padded sequences
    x_padded = torch.zeros(num_sequences, max_length)
    for ii in range(num_sequences):
        vec = to_vector(x[ii], tokens_to_idx)
        x_padded[ii, :orig_lengths[ii]] = vec

    x_padded = x_padded.to(torch.int64)
    return x_padded


