from load_teacher import *
import sys

corpus = sys.argv[1]

def main():
    print("beginning to test: " + corpus)

    if corpus == 'test':
        test_this(get_test)
    elif corpus == 'train':
        test_this(get_train)
    elif corpus == 'sample':
        test_this(load_sample)

    print("done testing (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
