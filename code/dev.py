"""
    Authors: Indiana Huey & Irmak Bukey
    Date: 21 April 2023

    This project implements a trigram language model with options for
    various smoothing techniques, namely
    - add-lambda
    - Good-Turing estimation
    - linear interpolation
    - absolute discounting
    - interpolated Kneser-Ney
    - Katz

    The smoothing techniques were implemented following their descriptions in the 1999 paper "An Empirical 
    Study of Smoothing Techniques for Language Modeling" by Chen and Goodman, except for absolute discounting, 
    which seems to have some ambiguity in the paper, and was instead implemented following lectures by Prof. Dave Kauchak.

    [1] Chen & Goodman. https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf
"""

from sys import argv
from math import log10

class Trigram_LM_Model:
    """
        A class representing a trigram language model with options for
        various smoothing technqiues, namely
        - add-lambda
        - Good-Turing estimation
        - linear interpolation
        - absolute discounting
        - interpolated Kneser-Ney
        - Katz
    """

    def __init__(self, train_filename, vocab_filename):
        """ 
            Create and train a model, given
            - the name of a vocabulary file
            - the name of a training file
        """
        with open(vocab_filename) as f:
            self.read_vocab(f)

        with open(train_filename) as f:
            self.train(f)


    def read_vocab(self, vocab_file):
        """
            Create a vocabulary set given,
            - the name of a vocabulary file
        """
        self.vocab = set(line for line in vocab_file)
        

    def train(self, train_file):
        """
            Train the model, given
            - the name of a training file
        """
        self.count_ngrams(train_file)

        self.get_nr_counts()

        # self.abc_alphas = {}
        # self.bc_alphas = {}

        self.abx_total_counts = {}
        self.bx_total_counts = {}


    def count_ngrams(self, train_file):
        """
            Count trigrams, bigrams, and unigrams in a dataset, given
            - the name of a training file
        """
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
        """
            Count trigram frequencies, as
            used in Good-Turing estimation (values of Nr)
        """
        self.nr = {}

        for a in self.trigram_counts:
            for b in self.trigram_counts[a]:
                for c in self.trigram_counts[a][b]:
                    r = self.trigram_counts[a][b][c]
                    self.nr[r] = 1 + self.nr.get(r, 0)


    def perplexity(self, test_filename, smoothing_technique, *parameters):
        """
            Compute the perplexity on test sentences, given
            - the name of a test file
            - the name of a smoothing technique
            - the values of parameters associated with the smoothing technique
        """
        total_log_prob = 0
        total_trigrams = 0

        with open(test_filename) as f:
            for line in f:
                words = ['<s>', '<ss>'] + line.split() + ['</ss>', '</s>']

                for i in range(2, len(words)):
                    a = words[i - 2] if words[i - 2] in self.vocab else '<unk>'
                    b = words[i - 1] if words[i - 1] in self.vocab else '<unk>'
                    c = words[i] if words[i] in self.vocab else '<unk>'

                    if smoothing_technique == 'add-lambda':
                        trigram_prob = self.__add_lambda(a, b, c, parameters[0])
                    elif smoothing_technique == 'good-turing':
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
        

    def __add_lambda(self, a, b, c, lambda_value):
        """ 
            Compute the probability of a trigram with add-lambda smoothing, given
            - a, the first word in the trigram
            - b, the second word
            - c, the third word
            - the value with which to smooth trigram counts
        """
        abc_count = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
        ab_count = self.bigram_counts.get(a, {}).get(b, 0)
        
        return (
            (abc_count + lambda_value) /
            (ab_count + (lambda_value * len(self.vocab)))
        )


    def __good_turing(self, a, b, c, max_threshold):
        """ 
            Compute the probability of a trigram with Good-Turing estimation, given
            - a, the first word in the trigram
            - b, the second word
            - c, the third word
            - the threshold under which to use Good-Turing estimates
        """
        r = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
        nr = self.nr[r]

        r_star = (r + 1) * ((nr + 1) / nr)

        # TODO: Not sure if N is correct. Confused by paper's notation.
        N = self.total_trigram_count

        ab_count = self.bigram_counts.get(a, {}).get(b, 0)

        return (
            r_star / N if r <= max_threshold
            else r / ab_count
        )


    def __linear_interpolation(self, a, b, c, trigram_weight, bigram_weight):
        """ 
            Compute the probability of a trigram with linear interpolation, given
            - a, the first word in the trigram
            - b, the second word
            - c, the third word
            - the ratio with which to weight the trigram probability
            - the ratio with which to weight the bigram probability
        """
        mle_abc_prob = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0) / self.bigram_counts.get(a, {}).get(b, 0)
        mle_bc_prob = self.bigram_counts.get(b, {}).get(c, 0) / self.unigram_counts.get(b, 0)
        mle_c_prob = self.unigram_counts.get(c, 0) / self.total_unigram_count

        return (
            (trigram_weight * mle_abc_prob) +
            (bigram_weight * mle_bc_prob) +
            ((1 - (trigram_weight + bigram_weight)) * mle_c_prob)
        )


    def __absolute_discounting(self, a, b, c, discount):
        """ 
            Compute the probability of a trigram with absolute discounting, given
            - a, the first word in the trigram
            - b, the second word
            - c, the third word
            - the absolute value with which to discount ngram counts
        """
        # TODO
        pass
            

    def __kneser_ney(self, a, b, c, discount):
        """ 
            Compute the probability of a trigram with interpolated Kneser-Ney smoothing, given
            - a, the first word in the trigram
            - b, the second word
            - c, the third word
            - the value with which to discount ngram counts
        """
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
            discounted_trigram_prob + 
            (reserved_trigram_mass * discounted_bigram_prob) + 
            (reserved_bigram_mass * mle_unigram_prob)
        )
    

    def __katz(self, a, b, c, ngram_weight):
        """ 
            Compute the probability of a trigram with Katz smoothing, given
            - a, the first word in the trigram
            - b, the second word
            - c, the third word
            - the ratio with which to weight ngram probabilities
        """
        # TODO
        abc_r = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
        bc_r = self.bigram_counts.get(b, {}).get(c, 0)
        c_r =  self.unigram_counts.get(c, 0)

        pass


    def tune_parameters(self, test_filename, smoothing_technique, parameter1_values, parameter2_values=[None]):
        """
            Tune the parameters of a smoothing technique, given,
            - the name of a test file
            - the name of a smoothing technique
            - an iterable of values for the first parameter
            - an iterable of values for the second parameter (optional)
        """
        best_perplexity = float('inf')
        best_v1 = None
        best_v2 = None

        for v1 in parameter1_values:
            for v2 in parameter2_values:
                perplexity = self.perplexity(test_filename, smoothing_technique, v1, v2)
                print(perplexity)

                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_v1 = v1
                    best_v2 = v2

        return best_v1, best_v2, best_perplexity
    
    
def main():
    train_filename = './data/dev'
    test_filename = './data/test'
    vocab_filename = './data/vocab'

    lambda_value = 0.1

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

    v1, v2, score = model.tune_parameters(test_filename, 'add-lambda', [0, 0.0000000000000000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])
    print(v1, v2, score)

    # print('add-lambda', model.perplexity(test_filename, 'add-lambda', lambda_value))
    # print('good turing', model.perplexity(test_filename, 'good-turing', good_turing_max))
    # print('linear interpolation', model.perplexity(test_filename, 'linear interpolation', trigram_weight, bigram_weight))
    # print('absolute discounting', model.perplexity(test_filename, 'absolute discounting', absolute_discount))
    # print('kneser-ney', model.perplexity(test_filename, 'kneser-ney', kneser_ney_discount))
    # print('katz', model.perplexity(test_filename, 'katz', katz_discount))

if __name__ == '__main__':
    main()
    