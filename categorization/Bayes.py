import collections
import math
import sys
import nltk


from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


# according to the number of file lists file,
# make train_doc_list as a dictionary, using classes as keys
# make test_doc_list as a list, without classes
def parse_arg():
    argv = sys.argv[1:]
    train_list = {}
    test_list = []
    N_doc = 0
    c_count = {}
    if len(argv) == 2:
        train_file = open(str(argv[0]), "r")
        test_file = open(str(argv[1]), "r")

        for line in train_file:
            if line.strip().split()[1] in train_list:
                train_list[line.strip().split()[1]].append(line.strip().split()[0])
                c_count[line.strip().split()[1]] += 1;
            else:
                train_list[line.strip().split()[1]] = [line.strip().split()[0]]
                c_count[line.strip().split()[1]] = 1;
            N_doc += 1

        for line in test_file:
            test_list.append(line.strip())

        train_file.close()
        test_file.close()

    elif len(argv) == 1:
        file = open(str(argv[0]), "r")
        count = 0
        for line in file:
            count += 1

        print('Total of "' + str(count) + '" files were given ...')
        train_num = input("How many files should be used as training documents? : ")

        file.seek(0)
        for line in file:
            if N_doc < train_num:
                if line.strip().split()[1] in train_list:
                    train_list[line.strip().split()[1]].append(line.strip().split()[0])
                else:
                    train_list[line.strip().split()[1]] = [line.strip().split()[0]]
                N_doc += 1
            else:
                test_list.append(line.strip().split()[0])
        file.close()

    else:
        print('usage: python SpacyLemma.py training_doc test_file')
        sys.exit(2)

    '''
    min_key = min(c_count, key = lambda k: c_count[k])
    for category in train_list:
        del train_list[category][c_count[min_key]:]
    '''
    return train_list, test_list, N_doc


def train_bayes(train_list, N_d):
    log_prior = {}
    V = []
    big_doc = {}
    log_likelihood = {}
    alpha = 0.05
    lc = LancasterStemmer()
    tags = ['NN', 'NNS', 'NNP', 'NNPS']
    for c in train_list.keys():
        # log_prior
        for file in train_list[c]:
            if c in log_prior:
                log_prior[c][0] = log_prior[c][0] + 1 / N_d
                log_prior[c][1] = math.log(log_prior[c][0])
            else:
                log_prior[c] = []
                log_prior[c].append(1 / N_d)
                log_prior[c].append(math.log(1 / N_d))

            # V & big_doc
            cur_file = open(file, 'r')
            token_file = sent_tokenize(cur_file.read())
            for sent in token_file:
                token_sent = word_tokenize(sent)
                tagged_sent = nltk.pos_tag(token_sent)
                for word in tagged_sent:
                    weight = 1;
                    if word[1] in tags:
                        weight = 3;
                    stem_word = lc.stem(word[0])
                    V.append(stem_word)
                    for i in range(weight):
                        if c in big_doc:
                            big_doc[c].append(stem_word)
                        else:
                            big_doc[c] = [stem_word]
            cur_file.close()

    # stop words
    stop_num = 50
    counter = collections.Counter(V)
    for stop_word, count in counter.most_common(stop_num):
        V = list(filter(lambda x: x != stop_word, V))
    V = list(set(V))

    # log_likelihood
    for c in train_list.keys():
        for w in V:
            log_likelihood[(w, c)] = math.log((big_doc[c].count(w) + alpha) / (len(big_doc[c]) + (alpha * len(V))))

    return log_prior, log_likelihood, V


def test_bayes(test_doc, log_prior, log_likelihood, C, V):
    sum = {}
    cur_file = open(test_doc, 'r')
    token_file = sent_tokenize(cur_file.read())
    lc = LancasterStemmer()
    for c in C:
        binaryNB = []
        sum[c] = log_prior[c][1]
        for sent in token_file:
            token_sent = word_tokenize(sent)
            tagged_sent = nltk.pos_tag(token_sent)
            for word in tagged_sent:
                stem_word = lc.stem(word[0])
                if stem_word not in binaryNB:
                    binaryNB.append(stem_word)
                    if stem_word in V:
                        sum[c] = sum[c] + log_likelihood[(stem_word, c)]
    C_NB = max(sum, key=sum.get)
    cur_file.close()
    return C_NB


if __name__ == '__main__':
    print('Parsing arguments ...')
    train_doc_list, test_doc_list, N = parse_arg()
    print('Training Naive Bayes ...')
    log_p, log_ll, v = train_bayes(train_doc_list, N)
    class_list = train_doc_list.keys()

    filename = input("Specify the output file name: ")
    outfile = open(filename, 'w')
    print('Testing Naive Bayes ...')
    for doc in test_doc_list:
        C_nb = test_bayes(doc, log_p, log_ll, class_list, v)
        result = doc + ' ' + C_nb + '\n'
        outfile.write(result)

    outfile.close()
