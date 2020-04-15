import sys


class Node:
    def __init__(self, key, left, right):
        self.left = left
        self.right = right
        self.key = key


def readCNF(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: python3 CKY.py <cnf_file>\n")
        sys.exit(1)
    else:
        try:
            cnf_file = open(str(argv[1]), 'r')
        except:
            sys.stderr.write("\nError: Could not open the CNF grammer: " + str(argv[1]) + "\n")
            sys.exit(1)

        CNF_1 = {}
        CNF_2 = {}

        for line in cnf_file:
            words = line.strip().split()
            if len(words) == 3:
                if words[2] not in CNF_1.keys():
                    CNF_1[words[2]] = set()
                CNF_1[words[2]].add(words[0])
            elif len(words) == 4:
                if (words[2], words[3]) not in CNF_2.keys():
                    CNF_2[(words[2], words[3])] = set()
                CNF_2[(words[2], words[3])].add(words[0])
        cnf_file.close()

    return CNF_1, CNF_2


def CKY(SENT, CNF_1, CNF_2):
    num = len(SENT)
    table = {}
    Complete = []
    # initialize 2d dictionary
    for i in range(num+1):
        table[i] = {}
        for j in range(num+1):
            table[i][j] = []

    # CKY algorithm
    for j in range(1, num+1):
        if SENT[j-1] in CNF_1.keys():
            for word in CNF_1[SENT[j-1]]:
                table[j-1][j].append(Node(word, Node(SENT[j-1], None, None), None))
        for i in range(j-2, -1, -1):
            for k in range(i+1, j):
                for B in table[i][k]:
                    for C in table[k][j]:
                        if (B.key, C.key) in list(CNF_2.keys()):
                            for word in CNF_2[(B.key, C.key)]:
                                table[i][j].append(Node(word, B, C))

    for node in table[0][num]:
        if node.key == 'S':
            Complete.append(node)

    return Complete


def traverse(node, parsed):
    if node.left:
        if parsed.endswith(']'):
            parsed = parsed + ' [' + node.key + ' '
        else:
            parsed = parsed + '[' + node.key + ' '
        parsed = traverse(node.left, parsed)
    else:
        parsed = parsed + node.key + ']'
    if node.right:
        parsed = traverse(node.right, parsed)
        parsed = parsed + ']'
    return parsed


def main(argv):
    cnf1, cnf2 = readCNF(argv)
    while True:
        sent = input('Enter Sentence to Parse: ')
        if sent == 'quit':
            break

        complete = CKY(sent.split(), cnf1, cnf2)

        if len(complete) == 0:
            sys.stderr.write("NO VALID PARSES\n")
            pass
        else:
            num = 1
            for node in complete:
                output = ""
                output = traverse(node, output)
                sys.stdout.write('Parse' + str(num) + ': ' + output + '\n')
                num += 1
            sys.stdout.write('Total of ' + str(len(complete)) + ' valid parses found ...\n\n')


if __name__ == "__main__":
    main(sys.argv)
