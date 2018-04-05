import os
import numpy as np
from helpers import PATH_TO_SAMPLES
from matplotlib import pyplot as plt
from string import punctuation
from copy import deepcopy
from nltk.corpus import stopwords

# useful globals
EOS_MARKERS = ['.', '!', '?']
MIN_SENTENCE_LENGTH = 2
STOPS = set(stopwords.words('english'))
PUNCT = set([c for c in punctuation] + ["''", '``', '...', '-rrb-', '-lrb-', '--'])

# useful helpers
def iter_sample_files():
    for fname in os.listdir(PATH_TO_SAMPLES):
        if fname.endswith('.story'):
            yield PATH_TO_SAMPLES + fname

def split_into_sentences(tokenized_text):
    sents = [[]]
    for i, token in enumerate(tokenized_text):
        sents[-1].append(token)
        if token in EOS_MARKERS and i != len(tokenized_text) - 1:
            sents.append([])
    return sents

def get_samples(clean=True):
    docs = []
    for f in iter_sample_files():
        a = Article(f)
        if clean:
            a.clean()
        docs.append(a)
    return docs

# useful class
class Article(object):
    def __init__(self, path_to_file):
        super(Article, self).__init__()
        self.summary_sents = []
        self.doc_sents = []
        self.extractive_preds = {}
        self.path = path_to_file
        self.is_clean = False

        # load data then deal with it
        lines = []
        with open(path_to_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        at_summ = False
        for i, line in enumerate(lines):
            if len(line) <= 1: # blank line
                continue
            elif line.startswith('@highlight'):
                at_summ = True
                continue

            #TODO: note that we are assuming CNN
            tok_txt = (line if i!=0 else line.split('-LRB- CNN -RRB- -- ')[-1]).split()
            if at_summ:
                self.summary_sents.append(tok_txt)
            else:
                for sentence in split_into_sentences(tok_txt):
                    if len(sentence) > 2: # sentence must be more than 2 tokens
                        self.doc_sents.append(sentence)

    def __len__(self):
        return len(self.doc_sents)

    def get_summary_string(self, newlines=False):
        c = '\n' if newlines else ' '
        return c.join([' '.join(x) for x in self.summary_sents])

    def get_doc_sent_string(self, idx):
        return ' '.join(self.doc_sents[idx])

    def get_fname(self):
        return self.path.split('/')[-1]

    def add_extraction_pred(self, extractor, extraction):
        self.extractive_preds[extractor] = extraction

    def serialize_extr_results(self, outdir):
        res_fname = ''.join([outdir, self.get_fname(), '.extr_res'])
        with open(res_fname, 'w', encoding='utf-8') as f:
            f.write('DOCUMENT:\n')
            for i in range(len(self)):
                f.write('{}: {}\n'.format(i, self.get_doc_sent_string(i)))
            f.write('\n=================================\n')
            f.write('Gold summary:\n')
            f.write(self.get_summary_string(newlines=True) + '\n')
            for ex_name, ex in self.extractive_preds.items():
                f.write('\n=================================\n')
                f.write('{} predicted the following:\n'.format(ex_name))
                f.write('\tSentences: {}\n'.format(ex.get_snum_str()))
                f.write('\tResults: \n{}\n'.format(ex.get_res_str()))
                f.write('\tSummary: \n{}\n'.format(ex.get_sum_str()))

    def clean(self, to_lower=True, remove_stops=True, remove_punct=True):
        cleaned_sents = []
        for s in self.summary_sents + self.doc_sents:
            sent = s
            if to_lower:
                sent = map(lambda x: x.lower(), sent)
            if remove_stops:
                sent = filter(lambda x: x not in STOPS and not "'" in x, sent)
            if remove_punct:
                sent = filter(lambda x: x not in PUNCT, sent)
            sent = list(sent)
            if len(sent) >= MIN_SENTENCE_LENGTH:
                cleaned_sents.append(sent)

        switch_idx = len(self.summary_sents)
        self.summary_sents = cleaned_sents[:switch_idx]
        self.doc_sents = cleaned_sents[switch_idx:]
        is_clean = True


class Extraction(object):
    def __init__(self, doc_fname, sent_nums, summary, results):
        super(Extraction, self).__init__()
        self.fname = doc_fname
        self.sents = sent_nums
        self.summ = summary
        self.rouge_res = results

    def get_res_str(self, tabs=2):
        s = []
        for res_name, res in self.rouge_res.items():
            s.append('{}{}: {:6.4f}'.format(tabs*'\t', res_name, res['r']))
        return '\n'.join(s)

    def get_sum_str(self, tabs=2):
        return '\n'.join(map(lambda s: '{}{}'.format(tabs*'\t', s), self.summ))

    def get_snum_str(self):
        return ', '.join(map(str, self.sents))

if __name__ == '__main__':
    articles = get_clean_samples()
