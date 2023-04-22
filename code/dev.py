"""
    Authors: Indiana Huey & Irmak Bukey
    Date: 21 April 2023

    Smoothing techniques.
"""

from sys import argv
from math import log10

class Trigram_LM_Model:

    def __init__(self, train_filename, vocab_filename, absolute_discount):
        self.absolute_discount = absolute_discount

        with open(vocab_filename) as f:
            self.read_vocab(f)

        with open(train_filename) as f:
            self.train(f)


    def read_vocab(self, vocab_file):
        self.vocab = set(line for line in vocab_file)
        

    def train(self, train_file):
        self.get_counts(train_file)


    def get_counts(self, train_file):
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.trigram_counts = {}

        self.total_unigram_count = 0

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

                self.total_unigram_count += 1

                encountered_words.add(words[i])
                a = b
                b = c


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

        if smoothing_technique == 'absolute discounting':
            # trigram-level terms
            abc_count = self.trigram_counts.get(a, {}).get(b, {}).get(c, 0)
            ab_count = self.bigram_counts.get(a, {}).get(b, 0)

            discounted_trigram_prob = (
                (abc_count - self.absolute_discount) /
                ab_count
            ) if ab_count > 0 else 0

            abx_unique = len(self.trigram_counts.get(a, {}).get(b, {}))
            abx_total = sum(self.trigram_counts.get(a, {}).get(b, {}).values())

            reserved_trigram_mass = (
                (abx_unique * self.absolute_discount) /
                abx_total
            )

            bc_count = self.bigram_counts.get(b, {}).get(c, 0)
            c_count = self.unigram_counts.get(c, 0)

            # bigram-level terms
            discounted_bigram_prob = (
                (bc_count - self.absolute_discount) /
                c_count
            ) if c_count > 0 else 0

            bx_unique = len(self.bigram_counts.get(b, {}))
            bx_total = sum(self.bigram_counts.get(b, {}).values())

            reserved_bigram_mass = (
                (bx_unique * self.absolute_discount) /
                bx_total
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

        return smoothed_prob


def main():
    train_filename = './data/train'
    test_filename = './data/test'
    vocab_filename = './data/vocab'
    absolute_discount = 0.1

    if absolute_discount < 0 or absolute_discount > 1:
        print('absolute discount must be in [0, 1]')
        return

    model = Trigram_LM_Model(train_filename, vocab_filename, absolute_discount)

    print(model.perplexity(test_filename, 'absolute discounting'))


if __name__ == '__main__':
    main()



        


