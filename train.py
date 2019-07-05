import os
import shutil
import subprocess
from zipfile import ZipFile


FILE_ALL_TARGET = './test_input/all_targets.txt'
FILE_TARGET = './test_input/targets.txt'
FOLDER_TARGETS = './test_input/target_corpora'
FOLDER_OUTPUT = './test_input/output'
ZIP_FILE = './results.zip'

N = 200


def store_output(words):
    with ZipFile(ZIP_FILE, 'a') as z:
        for word in words:
            path = FOLDER_OUTPUT + '/' + word
            z.write(path + '/output.dat', word + '.dat')
            shutil.rmtree(path)


def train_SCAN():
    with open(os.devnull, 'w') as FNULL:
        print('\tCreating corpora...')
        subprocess.call(['./tracking-meaning',
                         '-parameter_file=parameters.txt',
                         '-create_corpus', '-store=true'])

        print('\tTraining model...')
        subprocess.call(['./tracking-meaning',
                         '-parameter_file=parameters.txt', '-store=true'])
        print()


def transfer_lines(in_file, out_file, n):
    with open(in_file) as in_f:
        with open(out_file, 'w') as out_f:
            in_f.write(out_f.readlines(n))


if __name__ == '__main__':
    words = []
    with open(FILE_ALL_TARGET) as in_file:
        words = in_file.read().split('\n')

    i = 0
    while words:
        n = N
        i += 1
        shutil.rmtree(FOLDER_TARGETS)
        os.mkdir(FOLDER_TARGETS)

        words_chunck = []
        if len(words) < n:
            words, words_chunck = [], words
        else:
            words, words_chunck = words[n:], words[:n]

        print('Batch #' + str(i) + ',', len(words), 'words left.',
              'Top word of the batch is "' + words_chunck[0] + '".')

        with open(FILE_TARGET, 'w') as out_file:
            out_file.write('\n'.join(words_chunck))

        train_SCAN()
        store_output(words_chunck)
