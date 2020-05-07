import sys
from nltk.corpus import stopwords
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

training_portion = 0.8
dict_size = 5000
max_len = 300
embed_D = 64
epochs_num = 10


def parse_arg():
    argv = sys.argv[1:]
    train_set = {}
    test_set = set()

    train_files = open(str(argv[0]), 'r')
    test_files = open(str(argv[1]), 'r')
    for line in train_files:
        words = line.strip().split()
        if words[1] not in train_set:
            train_set[words[1]] = []
        train_set[words[1]].append(words[0])
    for line in test_files:
        words = line.strip().split()
        test_set.add((words[0], words[1]))

    return train_set, test_set


def verify(train_set):
    doc_num = []
    print('From the given training set, ' + str(len(train_set)) + ' labels were found')
    print('Number of training documents for each label:')
    for l in train_set:
        print(l + ': ' + str(len(train_set[l])))
        doc_num.append((l, len(train_set[l])))

    return doc_num


def load_text(train_set):
    doc_num = verify(train_set)
    doc_num = list(map(lambda x: (x[0], int(x[1] * training_portion)), doc_num))
    STOPWORDS = set(stopwords.words('english'))
    doc_total = {}

    train_total = []
    validate_total = []

    for l in train_set:
        doc_total[l] = []
        for file in train_set[l]:
            cur_file = open(file, 'r')
            file_content = " ".join((cur_file.read().lower()).split())
            for word in STOPWORDS:
                sw = ' ' + word + ' '
                file_content = file_content.replace(sw, ' ')
            doc_total[l].append(file_content)

    for c in doc_num:
        count = 0
        for content in doc_total[c[0]]:
            if count < c[1]:
                train_total.append((content, c[0]))
                count += 1
            else:
                validate_total.append((content, c[0]))

    random.shuffle(train_total)
    random.shuffle(validate_total)
    train_docs = list(map(lambda x: x[0], train_total))
    train_labels = list(map(lambda x: x[1], train_total))
    validate_docs = list(map(lambda x: x[0], validate_total))
    validate_labels = list(map(lambda x: x[1], validate_total))

    return train_docs, train_labels, validate_docs, validate_labels


def preprocessing(train_docs, train_labels, validate_docs, validate_labels):
    tokenizer1 = Tokenizer(num_words=dict_size, oov_token='<NF>')
    tokenizer1.fit_on_texts(train_docs)
    train_docs_seq = tokenizer1.texts_to_sequences(train_docs)
    train_pad = pad_sequences(train_docs_seq, maxlen=max_len, padding='post', truncating='post')
    validate_docs_seq = tokenizer1.texts_to_sequences(validate_docs)
    validate_pad = pad_sequences(validate_docs_seq, maxlen=max_len, padding='post', truncating='post')

    tokenizer2 = Tokenizer()
    tokenizer2.fit_on_texts(train_labels)
    train_labels_seq = np.array(tokenizer2.texts_to_sequences(train_labels))
    validate_labels_seq = np.array(tokenizer2.texts_to_sequences(validate_labels))

    return train_pad, validate_pad, train_labels_seq, validate_labels_seq


def RNN_train(t_pad, v_pad, t_labels_seq, v_labels_seq):
    # TODO activation = softmax / relu / tanh / sigmoid / hard_sigmoid / exponential / linear
    # TODO embed_D
    # TODO last Dense layer 6 -> 5
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(dict_size, embed_D),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embed_D)),
        tf.keras.layers.Dense(embed_D, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.summary()
    # TODO batch size?
    # TODO loss =
    # TODO optimizer = sgd / RMSprp / Adagrad/ Adadelta / Adam / Adamax / Nadam
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(t_pad, t_labels_seq, epochs=epochs_num, validation_data=(v_pad, v_labels_seq), verbose=1)

    return model

def RNN_test(model):
    print('hello')


def main():
    train_set, test_set = parse_arg()
    train_docs, train_labels, validate_docs, validate_labels = load_text(train_set)
    t_pad, v_pad, t_labels_seq, v_labels_seq = preprocessing(train_docs, train_labels, validate_docs, validate_labels)
    model = RNN_train(t_pad, v_pad, t_labels_seq, v_labels_seq)
    RNN_test(model)

if __name__ == '__main__':
    main()
