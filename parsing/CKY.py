import sys


def readCNF(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: python3 CKY.py <cnf_file>\n")
        sys.exit(1)
    else:
        try:
            cnf_file = open(str(argv[1]), 'r')
        except:
            sys.stderr.write("\nError: Could not open input file: " + str(argv[1]) + "\n")
            sys.exit(1)

        CNF_1 = {}
        CNF_2 = {}

        for line in cnf_file:
            words = line.strip().split()
            if len(words) == 3:
                if words[0] not in CNF_1.keys():
                    CNF_1[words[0]] = []
                CNF_1[words[0]].append(words[2])
            elif len(words) == 4:
                if words[0] not in CNF_2.keys():
                    CNF_2[words[0]] = []
                CNF_2[words[0]].append([words[2], words[3]])
        cnf_file.close()

    return CNF_1, CNF_2


def main(argv):
    cnf1, cnf2 = readCNF(argv)


if __name__ == "__main__":
    main(sys.argv)
