"""
    Authors: Indiana Huey & Irmak Bukey
    Date: 21 April 2023

    Smoothing techniques.

    Different versions of the same paper:
    - https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf
    - https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf

    Both versions are ambiguous in their desciptions of absolute discounting and Kneser-Ney smoothing.
    We implement absolute discounting as a backoff model and Kneser-Ney as an interpolated model.
"""

from sys import argv
from math import log10

class Trigram_LM_Model:

    def __init__(self, train_filename, vocab_filename):
        with open(vocab_filename) as f:
            self.read_vocab(f)

        with open(train_filename) as f:
            self.train(f)


    def read_vocab(self, vocab_file):
        self.vocab = set(line for line in vocab_file)
        

    def train(self, train_file):
        self.count_ngrams(train_file)

        self.get_nr_counts()

        # self.abc_alphas = {}
        # self.bc_alphas = {}

        self.abx_total_counts = {}
        self.bx_total_counts = {}


    def count_ngrams(self, train_file):
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.trigram_counts = {}

        self.total_unigram_count = 0
        self.total_trigram_count = 0

        encountered_words = set(['<s>', '<ss>', '<unk>', '</ss>', '</s>'])

        for line in train_file:
            words = ['<s>', '<ss>'] + line.split() + ['</ss>', '</s>']
            a = words[0]
            b = words[1]
            
            for i in range(2, len(words)):
                c = words[i] if words[i] in encountered_words else '<unk>'

                if a not in self.bigram_counts:
                    self.bigram_counts[a] = {}
                if a not in self.trigram_counts:
                    self.trigram_counts[a] = {}
                if b not in self.trigram_counts[a]:
                    self.trigram_counts[a][b] = {}

                self.unigram_counts[c] = 1 + self.unigram_counts.get(c, 0)
                self.bigram_counts[a][b] = 1 + self.bigram_counts[a].get(b, 0)
                self.trigram_counts[a][b][c] = 1 + self.trigram_counts[a][b].get(c, 0)

                # TODO: one of these must be wrong...
                self.total_unigram_count += 1
                self.total_trigram_count += 1

                encountered_words.add(words[i])
                a = b
                b = c


    def get_nr_counts(self):
        self.nr = {}

        for a in self.trigram_counts:
            for b in self.trigram_counts[a]:
                for c in self.trigram_counts[a][b]:
                    r = self.trigram_counts[a][b][c]
                    self.nr[r] = 1 + self.nr.get(r, 0)


    def perplexity(self, test_sentences, smoothing_technique, *parameters):
        total_log_prob = 0
        total_trigrams = 0

        with open(test_sentences) as f:
            for line in f:
                words = ['<s>', '<ss>'] + line.split() + ['</ss>', '</s>']

                for i in range(2, len(words)):
                    a = words[i - 2] if words[i - 2] in self.vocab else '<unk>'
                    b = words[i - 1] if words[i - 1] in self.vocab else '<unk>'
                    c = words[i] if words[i] in self.vocab else '<unk>'

                    if smoothing_technique == 'good-turing':
                        trigram_prob = self.__good_turing(a, b, c, parameters[0])
                    elif smoothing_technique == 'linear interpolation':
                        trigram_prob = self.__linear_interpolation(a, b, c, parameters[0], parameters[1])
                    elif smoothing_technique == 'absolute discounting':
                        trigram_prob = self.__absolute_discounting(a, b, c, parameters[0])
                    elif smoothing_technique == 'kneser-ney':
                        trigram_prob = self.__kneser_ney(a, b, c, parameters[0])
                    elif smoothing_technique == 'katz':
                        trigram_prob = self.__katz(a, b, c, parameters[0])

                    total_log_prob += log10(trigram_prob)

                total_trigrams += len(words)

        return 10 ** ((-1) * (total_log_prob / total_trigrams))    
        

    def __good_turing(self, a, b, c, max_threshold):
        r = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
        nr = self.nr[r]

        r_star = (r + 1) * ((nr + 1) / nr)

        # TODO: Not sure if N is correct. Confused by paper's notation.
        N = self.total_trigram_count

        if r <= max_threshold:
            return r_star / N
        else:
            ab_count = self.bigram_counts.get(a, {}).get(b, 0)
            return r / ab_count


    def __linear_interpolation(self, a, b, c, trigram_weight, bigram_weight):
        mle_trigram_prob = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0) / self.bigram_counts.get(a, {}).get(b, 0)
        mle_bigram_prob = self.bigram_counts.get(b, {}).get(c, 0) / self.unigram_counts.get(b, 0)
        mle_unigram_prob = self.unigram_counts.get(c, 0) / self.total_unigram_count

        return (
            (trigram_weight * mle_trigram_prob) +
            (bigram_weight * mle_bigram_prob) +
            ((1 - (trigram_weight + bigram_weight)) * mle_unigram_prob)
        )


    def __absolute_discounting(self, a, b, c, discount):
        # TODO
        pass
            

    def __kneser_ney(self, a, b, c, discount):
        # trigram-level terms
        abc_count = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
        ab_count = self.bigram_counts.get(a, {}).get(b, 0)

        discounted_trigram_prob = (
            (abc_count - discount) /
            ab_count
        ) if ab_count > 0 else 0

        abx_unique_count = len(self.trigram_counts.get(a, {}).get(b, {}))

        if (a, b) in self.abx_total_counts:
            abx_total_count = self.abx_total_counts[(a, b)]
        else:
            abx_total_count = sum(self.trigram_counts.get(a, {}).get(b, {}).values())
            self.abx_total_counts[(a, b)] = abx_total_count

        reserved_trigram_mass = (
            (abx_unique_count * discount) /
            abx_total_count
        )

        # bigram-level terms
        bc_count = self.bigram_counts.get(b, {}).get(c, 0)
        c_count = self.unigram_counts.get(c, 0)

        discounted_bigram_prob = (
            (bc_count - discount) /
            c_count
        ) if c_count > 0 else 0

        bx_unique_count = len(self.bigram_counts.get(b, {}))

        if b in self.bx_total_counts:
            bx_total_count = self.bx_total_counts[b]
        else:
            bx_total_count = sum(self.bigram_counts.get(b, {}).values())
            self.bx_total_counts[b] = bx_total_count

        reserved_bigram_mass = (
            (bx_unique_count * discount) /
            bx_total_count
        )

        # unigram_level terms
        mle_unigram_prob = (
            c_count /
            self.total_unigram_count
        )

        return (
            discounted_trigram_prob + (reserved_trigram_mass *
                discounted_bigram_prob + (reserved_bigram_mass *
                    mle_unigram_prob
                )
            )
        )
    

    def __katz(self, a, b, c, ngram_weight):
        # TODO
        trigram_r = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
        
        if trigram_r > 0:
            katz_trigram_count = trigram_r * ngram_weight
        else:
            pass


def main():
    train_filename = './data/dev'
    test_filename = './data/test'
    vocab_filename = './data/vocab'

    good_turing_max = 5

    trigram_weight = 0.6
    bigram_weight = 0.3

    absolute_discount = 0.2

    kneser_ney_discount = 0.2

    katz_discount = 0.2

    if trigram_weight + bigram_weight > 1:
        print('trigram weight + bigram weight must be less than or equal to 1')
        return

    if absolute_discount < 0 or absolute_discount > 1:
        print('absolute discount must be in [0, 1]')
        return

    model = Trigram_LM_Model(train_filename, vocab_filename)

    print('good turing', model.perplexity(test_filename, 'good-turing', good_turing_max))
    print('linear interpolation', model.perplexity(test_filename, 'linear interpolation', trigram_weight, bigram_weight))
    # print('absolute discounting', model.perplexity(test_filename, 'absolute discounting', absolute_discount))
    print('kneser-ney', model.perplexity(test_filename, 'kneser-ney', kneser_ney_discount))
    # print('katz', model.perplexity(test_filename, 'katz', katz_discount))

if __name__ == '__main__':
    main()
    