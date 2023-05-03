"""
    Authors: Indiana Huey & Irmak Bukey
    Date: 21 April 2023

    This project implements a trigram language model with options for
    various smoothing techniques, namely
    - add-lambda
    - linear interpolation
    - absolute discounting
    - interpolated Kneser-Ney

    The smoothing techniques were implemented following their descriptions in the 1999 paper "An Empirical 
    Study of Smoothing Techniques for Language Modeling" by Chen and Goodman, except for absolute discounting, 
    which seems to have some ambiguity in the paper, and was instead implemented following lectures by Prof. Dave Kauchak.

    [1] Chen & Goodman. https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf
"""

from math import log10

class Trigram_LM_Model:
    """
        A class representing a trigram language model with options for
        various smoothing technqiues, namely
        - add-lambda
        - linear interpolation
        - absolute discounting
        - interpolated Kneser-Ney
    """

    def __init__(self, train_filename):
        """ 
            Create and train a model, given
            - the name of a training file
        """
        with open(train_filename) as f:
            self.count_ngrams(f)

        # all words that occur more than once during training
        self.vocab = set(self.unigram_counts.keys())
        

    def count_ngrams(self, train_file):
        """
            Count trigrams, bigrams, and unigrams in a dataset, given
            - the name of a training file
        """
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.trigram_counts = {}

        self.total_unigram_count = 0
        self.total_bigram_count = 0
        self.total_trigram_count = 0

        encountered_words = set(['<s>', '<unk>', '</s>'])

        for line in train_file:
            words = ['<s>'] + line.split() + ['</s>']
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

                self.unigram_counts[a] = 1 + self.unigram_counts.get(a, 0)
                self.bigram_counts[a][b] = 1 + self.bigram_counts[a].get(b, 0)
                self.trigram_counts[a][b][c] = 1 + self.trigram_counts[a][b].get(c, 0)

                self.total_unigram_count += 1
                self.total_bigram_count += 1
                self.total_trigram_count += 1

                encountered_words.add(words[i])
                a = b
                b = c

            # count unigrams at end of sentence
            self.unigram_counts[a] = 1 + self.unigram_counts.get(a, 0)
            self.unigram_counts[b] = 1 + self.unigram_counts.get(a, 0)
            self.total_unigram_count += 2

            # count bigram at end of sentence
            if a not in self.bigram_counts:
                self.bigram_counts[a] = {}

            self.bigram_counts[a][b] = 1 + self.bigram_counts[a].get(b, 0)
            self.total_bigram_count += 1


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
                words = ['<s>'] + line.split() + ['</s>']

                for i in range(2, len(words)):
                    a = words[i - 2] if words[i - 2] in self.vocab else '<unk>'
                    b = words[i - 1] if words[i - 1] in self.vocab else '<unk>'
                    c = words[i] if words[i] in self.vocab else '<unk>'

                    if smoothing_technique == 'add-lambda':
                        trigram_prob = self.__add_lambda(a, b, c, parameters[0])
                    elif smoothing_technique == 'linear interpolation':
                        trigram_prob = self.__linear_interpolation(a, b, c, parameters[0], parameters[1])
                    elif smoothing_technique == 'absolute discounting':
                        trigram_prob = self.__absolute_discounting(a, b, c, parameters[0])
                    elif smoothing_technique == 'kneser-ney':
                        trigram_prob = self.__kneser_ney(a, b, c, parameters[0])

                    total_log_prob += log10(trigram_prob)

                total_trigrams += len(words) - 2

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


    def __linear_interpolation(self, a, b, c, trigram_weight, bigram_weight):
        """ 
            Compute the probability of a trigram with linear interpolation, given
            - a, the first word in the trigram
            - b, the second word
            - c, the third word
            - the ratio with which to weight the trigram probability
            - the ratio with which to weight the bigram probability
        """
        mle_abc_prob = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0) / self.bigram_counts.get(a, {}).get(b, 0) if self.bigram_counts.get(a, {}).get(b, 0) else 0
        mle_bc_prob = self.bigram_counts.get(b, {}).get(c, 0) / self.unigram_counts.get(b, 0) if self.unigram_counts.get(b, 0) else 0
        mle_c_prob = self.unigram_counts.get(c, 0) / self.total_unigram_count

        unigram_weight = 1 - trigram_weight - bigram_weight

        return (
            (trigram_weight * mle_abc_prob) +
            (bigram_weight * mle_bc_prob) +
            (unigram_weight * mle_c_prob)
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
        abx_unique_count = len(self.trigram_counts.get(a, {}).get(b, {}))

        discounted_trigram_prob = (
            max(abc_count - discount, 0) /
            ab_count
        ) if ab_count > 0 else 0

        reserved_trigram_mass = (
            (abx_unique_count * discount) /
            ab_count
        ) if ab_count else 0

        # bigram-level terms
        bc_count = self.bigram_counts.get(b, {}).get(c, 0)
        b_count = self.unigram_counts.get(b, 0)
        bx_unique_count = len(self.bigram_counts.get(b, {}))

        discounted_bigram_prob = (
            max(bc_count - discount, 0) /
            b_count
        ) if b_count > 0 else 0

        reserved_bigram_mass = (
            (bx_unique_count * discount) /
            b_count
        ) if b_count else 0

        # unigram_level terms
        c_count = self.unigram_counts.get(c, 0)

        mle_unigram_prob = (
            c_count /
            self.total_unigram_count
        )

        # interpolate entirely to lower-order models
        # if no information is provided by the trigram
        reserved_trigram_mass = reserved_trigram_mass if discounted_trigram_prob else 1

        return (
            discounted_trigram_prob + reserved_trigram_mass * (
                discounted_bigram_prob + reserved_bigram_mass * (
                    mle_unigram_prob
                )
            )
        )

    def tune_parameters(self, test_filename, smoothing_technique, parameter1_values, parameter2_values=[]):
        """
            Tune the parameters of a smoothing technique, given,
            - the name of a test file
            - the name of a smoothing technique
            - an iterable of values for the first parameter
            - an iterable of values for the second parameter (optional)
        """
        best_perplexity = float('inf')
        best_parameter1_value = None
        best_parameter2_value = None

        if parameter2_values:
            for parameter1_value, parameter2_value in zip(parameter1_values, parameter2_values):
                perplexity = self.perplexity(test_filename, smoothing_technique, parameter1_value, parameter2_value)

                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_parameter1_value = parameter1_value
                    best_parameter2_value = parameter2_value

            return best_parameter1_value, best_parameter2_value, best_perplexity
        
        else:
            for parameter1_value in parameter1_values:
                perplexity = self.perplexity(test_filename, smoothing_technique, parameter1_value)

                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_parameter1_value = parameter1_value

            return best_parameter1_value, best_perplexity
    
    
def main():
    train_filename = './data/train'
    test_filename = './data/dev'

    model = Trigram_LM_Model(train_filename)

    lambda_value, lambda_perplexity = model.tune_parameters(test_filename, 'add-lambda', (1*10**(-i) for i in range(10)))
    trigram_weight, bigram_weight, linear_inter_perplexity = model.tune_parameters(test_filename, 'linear interpolation', (.8, .7, .6, .5, .4, .1, .3), (.1, .2, .3, .4, .3, .8, .2))
    # absolute_discount, absolute_dis_perplexity = model.tune_parameters(test_filename, 'absolute discounting', (.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99))
    kn_discount, kn_perplexity = model.tune_parameters(test_filename, 'kneser-ney', (.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99))

    print(f'add-lambda\t{lambda_value}\t{lambda_perplexity}')
    print(f'linear interpolation\t{trigram_weight}\t{bigram_weight}\t{linear_inter_perplexity}')
    # print(f'absolute discounting\t{absolute_discount}\t{absolute_dis_perplexity}')
    print(f'kneser-ney\t{kn_discount}\t{kn_perplexity}')

if __name__ == '__main__':
    main()
    