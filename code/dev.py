"""
    Authors: Indiana Huey & Irmak Bukey
    Date: 21 April 2023

    Smoothing techniques.

    https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf
    https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf
"""

from sys import argv
from math import log10

class Trigram_LM_Model:

    def __init__(self, train_filename, vocab_filename, trigram_weight, bigram_weight, absolute_discount, katz_discount, good_turing_max):
        self.trigram_weight = trigram_weight
        self.bigram_weight = bigram_weight
        self.absolute_discount = absolute_discount
        self.katz_discount = katz_discount
        self.good_turing_max = good_turing_max

        with open(vocab_filename) as f:
            self.read_vocab(f)

        with open(train_filename) as f:
            self.train(f)


    def read_vocab(self, vocab_file):
        self.vocab = set(line for line in vocab_file)
        

    def train(self, train_file):
        self.get_counts(train_file)

        self.get_nr_counts()

        self.abc_alphas = {}
        self.bc_alphas = {}

        self.abx_total_counts = {}
        self.bx_total_counts = {}


    def get_counts(self, train_file):
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


    def perplexity(self, test_sentences, smoothing_technique):
        total_log_prob = 0
        total_trigrams = 0

        with open(test_sentences) as f:
            for line in f:
                words = ['<s>', '<ss>'] + line.split() + ['</ss>', '</s>']

                total_log_prob += self.log_prob(words, smoothing_technique)
                total_trigrams += len(words)

        return 10 ** ((-1) * (total_log_prob / total_trigrams))
    

    def log_prob(self, words, smoothing_technique):
        total_log_prob = 0

        for i in range(2, len(words)):
            a = words[i - 2]
            b = words[i - 1]
            c = words[i]

            prob = self.trigram_prob(a, b, c, smoothing_technique)
            total_log_prob += log10(prob)

        return total_log_prob
    

    def trigram_prob(self, a, b, c, smoothing_technique):
        a = a if a in self.vocab else '<unk>'
        b = b if b in self.vocab else '<unk>'
        c = c if c in self.vocab else '<unk>'

        if smoothing_technique == 'good-turing':
            r = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
            nr = self.nr[r]

            r_star = (r + 1) * ((nr + 1) / nr)

            # TODO: Not sure if N is correct. Confused by paper's notation.
            N = self.total_trigram_count

            if r <= self.good_turing_max:
                smoothed_prob = r_star / N
            else:
                ab_count = self.bigram_counts.get(a, {}).get(b, 0)
                smoothed_prob = r / ab_count


        elif smoothing_technique == 'linear interpolation':
            mle_trigram_prob = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0) / self.bigram_counts.get(a, {}).get(b, 0)
            mle_bigram_prob = self.bigram_counts.get(b, {}).get(c, 0) / self.unigram_counts.get(b, 0)
            mle_unigram_prob = self.unigram_counts.get(c, 0) / self.total_unigram_count

            smoothed_prob = (
                (self.trigram_weight * mle_trigram_prob) +
                (self.bigram_weight * mle_bigram_prob) +
                ((1 - (self.trigram_weight + self.bigram_weight)) * mle_unigram_prob)
            )

        elif smoothing_technique == False: #'absolute discounting':
            abc_count = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
            ab_count = self.bigram_counts.get(a, {}).get(b, 0)

            bc_count = self.bigram_counts.get(b, {}).get(c, 0)
            c_count = self.unigram_counts.get(c, 0)

            if abc_count > 0:
                smoothed_prob = (
                    (abc_count - self.absolute_discount) /
                    ab_count
                ) if ab_count > 0 else 0

            elif bc_count > 0:
                if (a, b, c) in self.abc_alphas:
                    abc_alpha = self.abc_alphas((a, b, c))
                else:
                    abc_alpha = self.compute_abc_alpha()
                    self.abc_alphas[(a, b, c)] = abc_alpha


                discounted_bigram_prob = (
                    (bc_count - self.absolute_discount) /
                    c_count
                ) if c_count > 0 else 0

                smoothed_prob = abc_alpha * discounted_bigram_prob

            else:
                if (a, b, c) in self.abc_alphas:
                    abc_alpha = self.abc_alphas((a, b, c))
                else:
                    pass


                if (b, c) in self.bc_alphas:
                    bc_alpha = self.bc_alphas((b, c))
                else:
                    pass


                mle_unigram_prob = (
                    c_count /
                    self.total_unigram_count
                )

                smoothed_prob = abc_alpha * bc_alpha * mle_unigram_prob



        # TODO: What... does this implement Absolute or Kneser-Ney?
        # TODO: I think this maybe implements un-modified Kneser-Ney
        # TODO: replace var self.absolute_discount
        elif smoothing_technique == 'kneser-ney':
            # trigram-level terms
            abc_count = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
            ab_count = self.bigram_counts.get(a, {}).get(b, 0)

            discounted_trigram_prob = (
                (abc_count - self.absolute_discount) /
                ab_count
            ) if ab_count > 0 else 0

            abx_unique_count = len(self.trigram_counts.get(a, {}).get(b, {}))

            if (a, b) in self.abx_total_counts:
                abx_total_count = self.abx_total_counts[(a, b)]
            else:
                abx_total_count = sum(self.trigram_counts.get(a, {}).get(b, {}).values())
                self.abx_total_counts[(a, b)] = abx_total_count

            reserved_trigram_mass = (
                (abx_unique_count * self.absolute_discount) /
                abx_total_count
            )

            # bigram-level terms
            bc_count = self.bigram_counts.get(b, {}).get(c, 0)
            c_count = self.unigram_counts.get(c, 0)

            discounted_bigram_prob = (
                (bc_count - self.absolute_discount) /
                c_count
            ) if c_count > 0 else 0

            bx_unique_count = len(self.bigram_counts.get(b, {}))

            if b in self.bx_total_counts:
                bx_total_count = self.bx_total_counts[b]
            else:
                bx_total_count = sum(self.bigram_counts.get(b, {}).values())
                self.bx_total_counts[b] = bx_total_count

            reserved_bigram_mass = (
                (bx_unique_count * self.absolute_discount) /
                bx_total_count
            )

            # unigram_level terms
            mle_unigram_prob = (
                c_count /
                self.total_unigram_count
            )

            smoothed_prob = (
                discounted_trigram_prob + (reserved_trigram_mass *
                    discounted_bigram_prob + (reserved_bigram_mass *
                        mle_unigram_prob
                    )
                )
            )

        elif smoothing_technique == 'katz':
            trigram_r = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
            
            if trigram_r > 0:
                katz_trigram_count = trigram_r * self.katz_discount
            else:
                pass

        else:
            print('invalid smoothing technique')
            smoothed_prob = 0

        return smoothed_prob


def main():
    train_filename = './data/train'
    test_filename = './data/test'
    vocab_filename = './data/vocab'
    trigram_weight = 0.6
    bigram_weight = 0.3
    absolute_discount = 0.2
    katz_discount = 0.2
    good_turing_max = 5

    if trigram_weight + bigram_weight > 1:
        print('trigram weight + bigram weight must be less than or equal to 1')
        return

    if absolute_discount < 0 or absolute_discount > 1:
        print('absolute discount must be in [0, 1]')
        return

    model = Trigram_LM_Model(train_filename, vocab_filename, trigram_weight, bigram_weight, absolute_discount, katz_discount, good_turing_max)

    print(model.perplexity(test_filename, 'kneser-ney'))
    print(model.perplexity(test_filename, 'good-turing'))
    print(model.perplexity(test_filename, 'linear interpolation'))



if __name__ == '__main__':
    main()



        


