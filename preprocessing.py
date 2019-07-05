import os
from os.path import expanduser
from time import sleep
from collections import Counter

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

import vlc

IN_PATH = "Corpuses/COHA/"
OUT_FILE = "./corpus.txt"
FILE_OCCURENCES = "./occurences.txt"
FILE_TARGETS = "./targets.txt"

TOP_WORDS = 50000  # The top n words that will be targets.


def create_targets(counter, file_targets, file_occ, top_words=TOP_WORDS):
    """ Create a txt file that contains all words that will be modelled. """
    with open(file_targets, 'w') as f_t:
        with open(file_occ, 'w') as f_o:
            for word, n in counter.most_common(top_words):
                f_t.write(word + '\n')
                f_o.write(word + ' ' + str(n) + '\n')


def get_year(filename):
    i_0 = filename.index("_") + 1
    i_1 = filename.index("_", i_0)
    return filename[i_0 : i_1]


def is_function_tag(tag):
    return tag in {"PRP", "PRP$", "TO", "WP", "WP$", "WRB", "UH", "PDT", "MD",
                   "EX", "DT", "CC", "IN", "RP", "WDT"}


def preprocess(f):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = []
    for sent in sent_tokenize(f.read()):
        words_and_tags = pos_tag(word_tokenize(sent))
        for word, tag in words_and_tags:

            word = word.lower()
            if len(word) > 1 and (word not in stop_words) and \
               not is_function_tag(tag) and word.isalnum():
                word = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                words.append(word.lower())

    return words


def get_wordnet_pos(treebank_tag):
    "https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python"
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Noun is the default in the lemmatizer.


def main():
    counter = Counter()

    with open(OUT_FILE, "w") as f_out:
        for folder in os.listdir(IN_PATH):
            if folder[:4] == "text":

                print("processing", folder, "...", flush=True)
                path = IN_PATH + folder + "/"
                for filename in os.listdir(path):

                    with open(path + filename) as f_in:
                        words = preprocess(f_in)
                        year = get_year(filename)
                        f_out.write(year + "\t" + ' '.join(words) + "\n")

                        counter.update(words)

    create_targets(counter, FILE_TARGETS, FILE_OCCURENCES)


if __name__ == "__main__":
    main()

    home = expanduser("~")
    mp = vlc.MediaPlayer(home + "/OneDrive/Docs/Code/Tools/Mob.mp3")
    mp.play()
    sleep(300)
    mp.stop()
