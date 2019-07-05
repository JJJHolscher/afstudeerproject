import os
from collections import defaultdict, Counter
import numpy as np
from math import log, sqrt
from matplotlib import pyplot as plt
import pandas as pd
import random
from scipy import stats
import seaborn as sns

PATH_CORPUS = './Frermann/main/test_input/corpus.txt'
FOLDER_GENUINE_OUTPUT = './genuine_output/'
FOLDER_SHUFFLED_OUTPUT = './shuffled_output/'
FREQUENCIES_GENUINE = './frequencies.txt'
FREQUENCIES_SHUFFLED = './shuffled_frequencies.txt'

SAMPLE_SIZE = 0

START_T = 1810
END_T = 2009
DELTA_T = 20

class DriftCalculator:
    """The class used for storing word senses, calculating polysemy and
    calculating semantic change.
    """

    def __init__(self, word, senses, occ=0, senses_avg=None):
        self.word = word  # The word for which the SCAN model is made.
        self.occ = occ  # Amount of times $self.word occured in the corpus.

        # $self.senses = [$sense_0, ..., $sense_K]
        # $sense = [$segment_0, ..., $segment_T]
        # segment = {'distr' : $distr, 'words' : $words, 'presence' : $presence}
        # $distr is the size of the segment compared to the other segments of
        # other senses.
        # $words is a list of the top-10 words in a segment.
        # $presence is a list of floats, each is the presence for a word in $words.
        self.senses, self.senses_avg = senses, senses_avg
        if type(senses) is str:
            self.senses, self.senses_avg = self.senses_from_file(senses)

        if senses is not None:
            self.K, self.T = self.senses.shape

        self.drift_funcs = {
            'individual_distr_diff' : self.individual_distr_diff,
            'individual_distr_var' : self.individual_distr_var,
            'individual_distr_word_diff' : self.individual_distr_word_diff,
            'change in word distributions' : self.individual_word_diff,
            'change in sense distributions' : self.KL_divergence_distr,
            'KS_test_distr' : self.KS_test_distr
        }

        self.polys_funcs = {
            'difference in word distributions' : self.word_distr_KL,
            'cluster coefficient' : self.cluster_coeff,
            'spearman' : self.spearman
        }

    @staticmethod
    def senses_from_file(filename):
        senses = []
        senses_avg = []

        with open(filename) as f:
            for _ in range(7):
                next(f)

            sense = []
            avg_distr = []
            for line in f:
                if line == "\n":
                    senses.append(np.array(sense))
                    avg_distr = []
                    sense = []
                    continue

                avg = False
                i = line.index(' ')
                distr, line = line[:i], line[i + 3:]
                time = {'words' : [], 'presence' : []}

                if distr == 'p(w|s)':
                    avg = True
                    time['distr'] = sum(avg_distr) / len(avg_distr)
                elif distr[0] == '=':
                    break
                else:
                    time['distr'] = float(distr)
                    avg_distr.append(float(distr))

                line = line[line.index(':') + 3:]

                i = line.find(';')
                while i != -1:
                    line, token = line[i + 2:], line[:i].split(' ')
                    time['words'].append(token[0])
                    time['presence'].append(float(token[1][1:-1]))
                    i = line.find(';')

                if avg:
                    senses_avg.append(time)
                else:
                    sense.append(time)

        return np.array(senses), np.array(senses_avg)

    def calc_drift(self, func_name, summarize=None):
        return self.drift_funcs[func_name](summarize=summarize)

    def calc_polysemy(self, func_name, dc_dict, summarize=None):
        return self.polys_funcs[func_name](dc_dict, summarize=summarize)

    # Methods for calculating drift
    def individual_distr_diff(self, summarize=np.mean):
        """ Calculate semantic drift by the shift of its sense distributions.

        The output drift is how much a single sense shifts in distribution
        in one time step on average.
        """

        scores = np.empty(self.T - 1)
        for sense_segments in self.senses:

            prev_distr = sense_segments[0]['distr']
            for t, segment in enumerate(sense_segments[1:-1]):
                distr = segment['distr']
                scores[t] = abs(prev_distr - distr)
                prev_distr = distr

        if summarize:
            return summarize(scores)
        return scores

    def individual_distr_var(self, summarize=np.mean):
        """ Calculate drift as the variance of the sense distributions. """

        scores = np.zeros(self.K)

        for k, sense_segments in enumerate(self.senses):
            avg_distr = self.senses_avg[k]['distr']
            sense_var = 0

            for segment in sense_segments:
                sense_var += (segment['distr'] - avg_distr)**2

            scores[k] = sense_var / len(sense_segments)

        if summarize:
            return summarize(np.log(scores))
        return np.log(scores)

    def individual_distr_word_diff(self, summarize=np.mean):
        """ Calculate drift by the shift in words per sense, normalized by
        the distribution of that sense.
        """

        scores = np.zeros(self.T - 1)

        for sense_segments in self.senses:
            for t, segment in enumerate(sense_segments[1:]):
                prev = sense_segments[t - 1]
                min_presence = min(segment['presence'])

                # We assume that any word not in the top-10 has a presense
                # equal to the lowest word in top-10 of the sense.
                for i_1, word in enumerate(segment['words']):
                    try:
                        i_2 = prev['words'].index(word)
                        prev_presence = prev['presence'][i_2]
                    except ValueError:
                        prev_presence = min_presence

                    presence = segment['presence'][i_1]
                    diff = abs(presence - prev_presence)
                    diff *= (prev['distr'] + segment['distr']) / 2
                    scores[t] += diff

        if summarize is not None:
            return summarize(scores)
        return scores

    def individual_word_diff(self, summarize=np.mean):
        scores = np.zeros(self.T - 1)

        for t in range(1, self.T):
            prev = self.senses[:, t - 1]

            for k, segment in enumerate(self.senses[:, t]):
                prev_segment = prev[k]

                min_presence = min(segment['presence'])
                d = defaultdict(lambda: min_presence)
                for i, w in enumerate(segment['words']):
                    d[w] = segment['presence'][i]

                word_distr = [d[w] for w in prev_segment['words']]

                scores[t - 1] += segment['distr'] * stats.entropy(word_distr, prev_segment['presence'])

        if summarize is not None:
            return np.log(summarize(scores))
        return np.log(scores)

    def KL_divergence_distr(self, prev=True, summarize=np.mean):
        if prev:
            range_obj = range(len(self.senses[0]) - 1)
        else:
            range_obj = range(len(self.senses[0]))

        scores = np.empty(len(range_obj))
        for t in range_obj:
            sense_distr = np.empty(self.K)
            other_distr = np.empty(self.K)

            for k in range(len(self.senses)):
                sense_distr[k] = self.senses[k][t]['distr']
                if prev:
                    other_distr[k] = self.senses[k][t - 1]['distr']
                else:
                    other_distr[k] = self.senses_avg[k]['distr']

            scores[t] = stats.entropy(sense_distr, other_distr)

        if summarize is not None:
            return np.log(summarize(scores))
        return np.log(scores)

    def KS_test_distr(self, prev=False, summarize=np.amax):
        scores = np.empty(self.T)

        other_distr = []
        if not prev:
            other_distr = [sense['distr'] for sense in self.senses_avg]

        for t in range(1, len(self.senses[0])):
            curr_distr = []
            if prev:
                other_distr = []

            for k in range(len(self.senses)):
                if prev:
                    other_distr.append(self.senses[k][t - 1]['distr'])
                curr_distr.append(self.senses[k][t]['distr'])

            scores[t] = stats.ks_2samp(other_distr, curr_distr)[0]

        if summarize is not None:
            return summarize(scores)
        return scores

    # Methods for calculating polysemy
    def word_distr_KL(self, dc_dict, summarize=np.amax):
        scores = np.empty(2 * self.K * (self.K - 1))
        score_index = 0

        for k_0, sense_0 in enumerate(self.senses_avg):

            min_presence = min(sense_0['presence'])
            d = defaultdict(lambda: min_presence)
            for i, w in enumerate(sense_0['words']):
                d[w] = sense_0['presence'][i]

            for k_1, sense_1 in enumerate(self.senses_avg):
                if k_0 == k_1: continue

                sense_0_presence = [d[w] for w in sense_1['words']]
                scores[score_index] = sense_0['distr'] * stats.entropy(sense_0_presence, sense_1['presence'])
                score_index += 1
                scores[score_index] = sense_1['distr'] * stats.entropy(sense_1['presence'], sense_0_presence)
                score_index += 1

        # scores > 10**-12 is used to prevent problems with the log transform.
        if summarize is not None:
            return summarize(np.log(scores[scores > 10**-12]))
        return np.log(scores[scores > 10**-12])

    def cluster_coeff(self, dc_dict, summarize=None):
        context_words = set()
        for sense in self.senses_avg:
            context_words.update(sense['words'])

        cluster, n = 0, 0
        for w_0 in context_words:
            if w_0 not in dc_dict: continue

            context_of_w_0 = set()
            for sense in dc_dict[w_0].senses_avg:
                context_of_w_0.update(sense['words'])

            for w_1 in context_words:
                if w_0 == w_1: continue

                if w_1 in context_of_w_0:
                    cluster += 1
                n += 1

        if n > 0:
            return -cluster / n
        return np.inf

    def spearman(self, dc_dict, summarize=np.amin):
        spearmans = []

        for i, sense_0 in enumerate(self.senses_avg):

            min_presense = min(sense_0['presence'])
            d = defaultdict(lambda: min_presense)
            for j, word in enumerate(sense_0['words']):
                d[word] = sense_0['presence'][j]

            for sense_1 in self.senses_avg[i + 1:]:
                word_distr = [d[word] for word in sense_1['words']]
                corr, _ = stats.spearmanr(word_distr, sense_1['presence'])
                spearmans.append(corr)

        if summarize is not None:
            return summarize(spearmans)
        return spearmans


def calc_drifts(dc_arr, drift_funcs, summaries):
    """Calculate semantic change with multiple measures.

    Arguments:
    dc_arr       An N array of DriftCalculators, one for each word in the sample
    drift_funcs  An D array of function names for calculating drift
    summaries    An D array of how the output of the drift should be summarized

    Returns a DxN array of calculated semantic change values.
    """

    drifts = np.array([np.empty(len(dc_arr)) for _ in drift_funcs])

    for i, dc in enumerate(dc_arr):
        for j, drift_func in enumerate(drift_funcs):
            drifts[j][i] = dc.calc_drift(drift_func, summarize=summaries[j])

        if (i + 1) % (len(dc_arr) // 10) == 0:
            print('.', end='', flush=True)
    print(' ', end='')

    return drifts


def calc_polys(dc_arr, polys_funcs, summaries):
    """Calculate the polysemy using various measures.

    Arguments:
    dc_arr       A N array of DriftCalculators, one for each word in the sample
    polys_funcs  A P array of function names for calculating polysemy
    summaries    A P array of how the output of the polysemy should be summarized

    Returns a PxN array of calculated polysemy values.
    """

    polys = np.array([np.empty(len(dc_arr)) for _ in polys_funcs])
    dc_dict = {dc.word : dc for dc in dc_arr}

    for i, dc in enumerate(dc_arr):
        for j, polys_func in enumerate(polys_funcs):
            score = dc.calc_polysemy(polys_func, dc_dict, summarize=summaries[j])
            polys[j][i] = score

        if (i + 1) % (len(dc_arr) // 10) == 0:
            print('.', end='', flush=True)
    print(' ', end='')

    return polys


def compare_with_hamilton(words, freqs, drifts_g, drifts_s, polys_g,
                          polys_s, drift_funcs, polys_funcs):
    """Show a qualitative analysis on Hamiltons findings."""

    print('sorting four measures', end='', flush=True)
    drifts_g_ranks = np.array([rank_arr(d) for d in drifts_g])
    print('.', end='', flush=True)
    drifts_s_ranks = np.array([rank_arr(d) for d in drifts_s])
    print('.', end='', flush=True)
    polys_g_ranks = np.array([rank_arr(p) for p in polys_g])
    print('.', end='', flush=True)
    polys_s_ranks = np.array([rank_arr(p) for p in polys_s])
    print('.')

    top_polys = np.array(['yet', 'always', 'even', 'little', 'called', 'also',
                          'sometimes', 'great', 'still', 'quite'])
    print('Hamiltons most polysemous words:')
    compare_with_hamilton_formatter(top_polys, words, polys_g_ranks,
                                    polys_s_ranks, polys_funcs)

    bottom_polys = np.array(['photocopying', 'retrieval', 'thirties', 'mom',
                             'sweater', 'forties', 'seventeenth', 'fifteenth',
                             'holster', 'postage'])
    print('Hamiltons least polysemous words:')
    compare_with_hamilton_formatter(bottom_polys, words, polys_g_ranks,
                                    polys_s_ranks, polys_funcs)

    ppmi_top_drift = np.array(['know', 'got', 'would', 'decided', 'think',
                               'stop', 'remember', 'started', 'must',
                               'wanted'])
    print('Hamiltons most drifting words, according to his ppmi:')
    compare_with_hamilton_formatter(ppmi_top_drift, words, drifts_g_ranks,
                                    drifts_s_ranks, drift_funcs)

    svd_top_drift = np.array(['harry', 'headed', 'calls', 'gay', 'wherever',
                              'male', 'actually', 'special', 'cover',
                              'naturally'])
    print('Hamiltons most drifting words, according to his svd:')
    compare_with_hamilton_formatter(svd_top_drift, words, drifts_g_ranks,
                                    drifts_s_ranks, drift_funcs)

    sgns_top_drift = np.array(['wanting', 'gay', 'check', 'starting', 'major',
                               'actually', 'touching', 'harry', 'headed',
                               'romance'])
    print('Hamiltons most drifting words, according to his word2vec:')
    compare_with_hamilton_formatter(sgns_top_drift, words, drifts_g_ranks,
                                    drifts_s_ranks, drift_funcs)


def compare_with_hamilton_formatter(hamil_words, words, ranks_g, ranks_s,
                                    funcs):
    ix = np.amax(np.array([words == h for h in hamil_words]), axis=0)
    measures, headers = [], []
    for i, func_name in enumerate(funcs):
        measures.extend([ranks_g[i][ix], ranks_s[i][ix]])
        headers.extend(['genuine rank of ' + func_name,
                        'shuffled rank of ' + func_name])
    print_words(words[ix], np.array(measures), np.array(headers))


def create_calculators(output_folder, top_words, appendix='.dat'):
    drift_calculators = np.empty(len(top_words), dtype=DriftCalculator)

    for i, word in enumerate(top_words):
        drift_calculators[i] = DriftCalculator(word, output_folder + word +
                                               appendix)

        if (i + 1) % (len(top_words) // 10) == 0:
            print('.', end='', flush=True)
    print(' ', end='')

    return drift_calculators


def get_top(measure, n=10, reverse=False):
    """Return the indices of the highest n measures."""

    top = np.arange(n)
    if not reverse:
        top = ~ (top + 1)

    return np.argsort(measure)[top]


def load_freq(filename, top_words):
    freq = dict()
    top_words_set = set(top_words)

    with open(filename) as f:
        # Each line consists of: 'word, freq_in_year_0, ..., freq_in_year_T'
        for line in f:
            line = line.split(' ')[:-1]
            word, line = line[0], line[1:]
            if word not in top_words_set: continue

            freqs_per_t = np.zeros((len(line) - 1) // DELTA_T + 1, dtype=int)
            for t, i in enumerate(range(0, len(line), DELTA_T)):
                for freq_in_year in line[i : min(i + DELTA_T, len(line))]:
                    freqs_per_t[t] += int(freq_in_year)

            freq[word] = freqs_per_t

    return freq


def make_plots_0(in_freqs_g, in_freqs_s, in_drifts_g, in_drifts_s, in_polys_g,
               in_polys_s, drift_funcs, polys_funcs):
    """Make hexabin plots that should show any laws of semantic change.

    Arguments:
    in_freq    N array of frequencies
    in_drifts  DxN array of drift values for each measure
    in_polys   PxN array of polysemy values for each measure
    funcs      D or P array containing the names of the measures
    """
    # Set up seaborn styles
    sns.set(style='whitegrid', palette='pastel', color_codes=True)
    sns.set_context('paper'); sns.set_style('ticks')

    for i, drift_func_name in enumerate(drift_funcs):
        i_g, i_s = np.isfinite(in_drifts_g[i]), np.isfinite(in_drifts_s[i])
        freqs_g, freqs_s = np.log(in_freqs_g[i_g]), np.log(in_freqs_s[i_s])
        drifts_g, drifts_s = in_drifts_g[i][i_g], in_drifts_s[i][i_s]

        # Draw plots for the law of conformity.
        grid = sns.jointplot(freqs_g, drifts_g, kind='hex', color='b')
        grid.set_axis_labels('log frequency', drift_func_name)
        plt.suptitle('law of conformity in the genuine corpus')
        plt.show()
        grid = sns.jointplot(freqs_s, drifts_s, kind='hex', color='r')
        grid.set_axis_labels('log frequency', drift_func_name)
        plt.suptitle('law of conformity in the shuffled corpus')
        plt.show()

        for j, polys_func_name in enumerate(polys_funcs):
            j_g, j_s = np.isfinite(in_polys_g[j]), np.isfinite(in_polys_s[j])
            j_g, j_s = j_g & i_g, j_s & i_s
            drifts_g, drifts_s = in_drifts_g[i][j_g], in_drifts_s[i][j_s]
            polys_g, polys_s = in_polys_g[j][j_g], in_polys_s[j][j_s]

            # Draw plots for the law of innovation.
            grid = sns.jointplot(polys_g, drifts_g, kind='hex', color='b')
            grid.set_axis_labels(polys_func_name + ' (polysemy)', drift_func_name)
            plt.suptitle('law of innovation in the genuine corpus')
            plt.show()
            grid = sns.jointplot(polys_s, drifts_s, kind='hex', color='r')
            grid.set_axis_labels(polys_func_name + ' (polysemy)', drift_func_name)
            plt.suptitle('law of innovation in the shuffled corpus')
            plt.show()


def make_plots(in_freqs_g, in_freqs_s, in_drifts_g, in_drifts_s, in_polys_g,
                 in_polys_s, drift_funcs, polys_funcs):
    """Make hexabin plots that should show any laws of semantic change.

    Arguments:
    in_freq    N array of frequencies
    in_drifts  DxN array of drift values for each measure
    in_polys   PxN array of polysemy values for each measure
    funcs      D or P array containing the names of the measures
    """
    # Set up seaborn styles
    sns.set(style='whitegrid', palette='pastel', color_codes=True)
    sns.set_context('paper'); sns.set_style('ticks')

    for i, drift_func_name in enumerate(drift_funcs):
        i_g, i_s = np.isfinite(in_drifts_g[i]), np.isfinite(in_drifts_s[i])
        freqs_g, freqs_s = np.log(in_freqs_g[i_g]), np.log(in_freqs_s[i_s])
        drifts_g, drifts_s = in_drifts_g[i][i_g], in_drifts_s[i][i_s]

        if drift_func_name == 'change in word distributions':
            ylim = (-5, -2)
        elif drift_func_name == 'change in sense distributions':
            ylim = (-6, -1)
        xlim = (7, 13)

        # Draw plots for the law of conformity.
        grid = sns.jointplot(freqs_g, drifts_g, kind='hex', color='b',
                             height=3.5, ratio=4,
                             xlim=xlim, ylim=ylim)
        plt.tight_layout()
        plt.savefig(drift_func_name + '_freq_genuine.pdf', bbox_inches='tight')

        grid = sns.jointplot(freqs_s, drifts_s, kind='hex', color='r',
                             height=3.5, ratio=4,
                             xlim=xlim, ylim=ylim)
        plt.tight_layout()
        plt.savefig(drift_func_name + '_freq_shuffled.pdf', bbox_inches='tight')

        for j, polys_func_name in enumerate(polys_funcs):
            j_g, j_s = np.isfinite(in_polys_g[j]), np.isfinite(in_polys_s[j])
            j_g, j_s = j_g & i_g, j_s & i_s
            drifts_g, drifts_s = in_drifts_g[i][j_g], in_drifts_s[i][j_s]
            polys_g, polys_s = in_polys_g[j][j_g], in_polys_s[j][j_s]

            if polys_func_name == 'cluster coefficient':
                xlim = (-1, 0)
            elif polys_func_name == 'difference in word distributions':
                xlim = (-7, -4)

            # Draw plots for the law of innovation.
            grid = sns.jointplot(polys_g, drifts_g, kind='hex', color='b',
                             height=3.5, ratio=4,
                             xlim=xlim, ylim=ylim)
            plt.tight_layout()
            plt.savefig(drift_func_name + '_' + polys_func_name +
                        '_genuine.pdf', bbox_inches='tight')
            grid = sns.jointplot(polys_s, drifts_s, kind='hex', color='r',
                             height=3.5, ratio=4,
                             xlim=xlim, ylim=ylim)
            plt.tight_layout()
            plt.savefig(drift_func_name + '_' + polys_func_name +
                        '_shuffled.pdf', bbox_inches='tight')


def print_words(words, measure_arr, column_headers):
    """Print the words and their statistics in a padded table.

    Arguments:
    words           An M array of words
    measure_arr     An NxM array of statistics assosiated to the words
    column_headers  An N array of headers for the measures
    """

    # Calculate the paddings for each column.
    padding_words = max([len(w) for w in words]) + 2

    paddings = np.array([len(h) + 2 for h in column_headers], dtype=int)
    for i, measure in enumerate(measure_arr):
        for value in measure:
            if len(str(value)) + 2 > paddings[i]:
                paddings[i] = len(str(value)) + 2

    # Insert the headings and paddings in the header line.
    string = ' ' * padding_words
    for i, header in enumerate(column_headers):
        padding = paddings[i] - len(header)
        string += header + ' ' * padding
    string += '\n'

    # Insert the values and paddings in the rest of the table.
    for i, word in enumerate(words):
        string += word + ' ' * (padding_words - len(word))
        for j, value in enumerate(measure_arr[:, i]):
            string += str(value) + ' ' * (paddings[j] - len(str(value)))
        string += '\n'

    print(string[:-1])


def qualitative_analysis(words, freqs, drifts_g=[], drifts_s=[], polys_g=[],
                         polys_s=[], drift_funcs=[], polys_funcs=[], n=10):
    """Show the most and least drifting and polysemous words.

    Arguments:
    words    An N array of words to be analysed
    freqs    An N array of word frequency
    drifts_  An DxN array where the columns are different measures for drift
             There is one for the genuine, and the shuffled corpus
    polys_   An PxN array where the columns are different measures for polysemy
             There is one for the genuine, and the shuffled corpus
    _funcs   The names of the measurements. There is one for drift and one for
             polysemy
    n        The number of words that are to be analysed.
    """

    # Analyse the most and least drifting words.
    for i, func_name in enumerate(drift_funcs):
        headers = np.array(['frequency', 'genuine ' + func_name,
                            'shuffled ' + func_name])
        #   Show the most drifting words
        top = get_top(drifts_g[i], n=n)
        measures_arr = np.array([freqs[top], drifts_g[i][top],
                                 drifts_s[i][top]])
        print('\nTop', n, 'words ranked by', func_name)
        print_words(words[top], measures_arr, headers)
        #   Show the least drifting words
        bottom = get_top(drifts_g[i], n=n, reverse=True)
        measures_arr = np.array([freqs[bottom], drifts_g[i][bottom],
                                 drifts_s[i][bottom]])
        print('\nBottom', n, 'words ranked by', func_name)
        print_words(words[bottom], measures_arr, headers)

    # Analyse the most and least polysemous words.
    for i, func_name in enumerate(polys_funcs):
        headers = np.array(['frequency', 'genuine ' + func_name,
                            'shuffled ' + func_name])
        #   Show most polysemous words
        top = get_top(polys_g[i], n=n)
        measures_arr = np.array([freqs[top], polys_g[i][top], polys_s[i][top]])
        print('\nTop', n, 'words ranked by', func_name)
        print_words(words[top], measures_arr, headers)
        #   Show least polysemous words
        bottom = get_top(polys_g[i], n=n, reverse=True)
        measures_arr = np.array([freqs[bottom], polys_g[i][bottom],
                                 polys_s[i][bottom]])
        print('\nBottom', n, 'words ranked by', func_name)
        print_words(words[bottom], measures_arr, headers)


def rank_arr(arr):
    """Ranks an array
    from https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
    """
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))
    return ranks


def show_avg_drift_per_corpus(dc_g, dc_s, drift_funcs):
    sns.set(style='whitegrid', palette='pastel', color_codes=True)

    for func_name in drift_funcs:
        d = {'time bins' : [], 'corpus' : [], func_name : []}
        years = [t * DELTA_T + START_T for t, _ in enumerate(dc_g[0].calc_drift(func_name))]

        for dc in dc_g:
            d['time bins'].extend(years)
            d['corpus'].extend(['Genuine' for _ in years])
            d[func_name].extend(dc.calc_drift(func_name))

        for dc in dc_s:
            d['time bins'].extend(years)
            d['corpus'].extend(['Shuffled' for _ in years])
            d[func_name].extend(dc.calc_drift(func_name))

        df = pd.DataFrame(d)

        # Draw a nested violinplot and split the violins for easier comparison
        sns.violinplot(x='time bins', y=func_name, hue='corpus', split=True,
                       inner='quart', palette={'Genuine': 'b', 'Shuffled': 'r'},
                       data=df)
        sns.despine(left=True)
        plt.show()


def trendline(xd, yd, order=1, c='r', alpha=1, plot_r=False, text_pos=None):
    """ Make a line of best fit

    This function is copied from Hamilton's git of his paper about statistical
    laws of semantic change: https://github.com/williamleif/histwords
    """

    # Calculate trendline
    coeffs = np.polyfit(xd, yd, order)

    intercept = coeffs[-1]
    slope = coeffs[-2]
    if order == 2:
        power = coeffs[0]
    else:
        power = 0

    minxd = np.min(xd)
    maxxd = np.max(xd)

    xl = np.array([minxd, maxxd])
    yl = power * xl ** 2 + slope * xl + intercept

    # Plot trendline
    plt.plot(xl, yl, color=c, alpha=alpha)

    # Calculate R Squared
    r = stats.pearsonr(xd, yd)[0]

    if plot_r == False:
        #Plot R^2 value
        if text_pos == None:
            text_pos = (0.9 * maxxd + 0.1 * minxd, 0.9 * np.max(yd) + 0.1 * np.min(yd),)
        plt.text(text_pos[0], text_pos[1], '$R = %0.2f$' % r)
    else:
        #Return the R^2 value:
        return r


if __name__ == "__main__":
    top_words = np.array([w[:-4] for w in os.listdir(FOLDER_GENUINE_OUTPUT)])
    if SAMPLE_SIZE:
        sample = top_words[np.random.choice(len(top_words), SAMPLE_SIZE)]
    else:
        sample = top_words

    print('loading frequencies...')
    genuine_freq = load_freq(FREQUENCIES_GENUINE, sample)
    shuffled_freq = load_freq(FREQUENCIES_SHUFFLED, sample)
    freqs_g = np.array([sum(genuine_freq[w]) for w in sample])
    freqs_s = np.array([sum(shuffled_freq[w]) for w in sample])

    print('loading word senses', end='')
    dc_g = create_calculators(FOLDER_GENUINE_OUTPUT, sample)
    dc_s = create_calculators(FOLDER_SHUFFLED_OUTPUT, sample)

    print('\ncalculating drift', end='')
    drift_funcs = ['change in word distributions',
                   'change in sense distributions']

    drift_summaries = [np.mean, np.mean]
    drifts_g = calc_drifts(dc_g, drift_funcs, drift_summaries)
    drifts_s = calc_drifts(dc_s, drift_funcs, drift_summaries)

    print('\ncalculating polysemy', end='')
    polys_funcs = ['cluster coefficient', 'difference in word distributions']
    polys_summaries = [np.mean, np.mean]
    polys_g = calc_polys(dc_g, polys_funcs, polys_summaries)
    polys_s = calc_polys(dc_s, polys_funcs, polys_summaries)
    print()

    if not SAMPLE_SIZE:
        compare_with_hamilton(sample, freqs_g, drifts_g, drifts_s, polys_g,
                              polys_s, drift_funcs, polys_funcs)
        qualitative_analysis(sample, freqs_g, drifts_g, drifts_s, polys_g,
                             polys_s, drift_funcs, polys_funcs, n=30)

    make_plots(freqs_g, freqs_s, drifts_g, drifts_s, polys_g, polys_s,
               drift_funcs, polys_funcs)

    print('calculating the average semantic change...')
    show_avg_drift_per_corpus(dc_g, dc_s, set(drift_funcs))


