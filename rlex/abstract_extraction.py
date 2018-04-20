import numpy as np
from abc import ABC, abstractmethod
from rouge import Rouge

# statics
NUM_SENTS_EXTRACT = 3

# helpers
def get_score(summary, reference, option=None):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference, avg=True)
    if option == 'mean':
        return np.mean([scores[key]['r'] for key in scores])
    elif option in scores.keys():
        return scores[option]['r']
    return scores

# parameters storage
class Params(object):
    """
    Parameters for an extractor class. Mostly RL parameters.
    """
    def __init__(self, **kwargs):
        # reminders of some parameters to use
        self.v_lr = None
        self.p_lr = None
        self.gamma = 1.
        self.use_baseline = True
        self.epsilon = 0.5
        self.schedule = lambda time: self.epsilon / time
        self.method = 'softmax'
        self.update_only_last = True
        self.opt_option = 'mean'
        self.non_default_params = set()
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            try:
                if self.__getattribute__(key) == value:
                    continue
            except AttributeError:
                pass
            setattr(self, key, value)
            self.non_default_params.add(key)

    def to_name(self):
        return '__'.join([f'{attr}-{self.__getattribute__(attr)}'
                          for attr in self.non_default_params])


# base extractor
class Extractor(ABC):
    """
    Base class for something that extracts from an article.
    Useful for abstracting away the essential aspects of a
    extractive summarizer (e.g., Lead3 vs PolicyGradient)
    """
    def __init__(self, name, params):
        self.name = name
        self.params = params

    @staticmethod
    def is_learner():
        return False

    def extract_from_articles(self, article_list):
        """
        :param article_list: List of Article
        :return: List of Extraction
        """
        extractions = []
        for a in article_list:
            extractions.append(self.extract_summary(a))
        return extractions

    def extract_summary(self, article, sentnums=None, **kwargs):
        """
        :param article: Article
        :param sentnums: option list of sentence numbers to get
        :return: Extraction
        """
        if sentnums is None:
            sentnums = self._extract_sentums(article, **kwargs)
        extraction = [article.get_doc_sent_string(j) for j in sentnums]
        ref = [article.get_summary_string()]
        fname = article.get_fname()
        result = get_score([' '.join(extraction)], ref)
        return Extraction(fname, sentnums, extraction, result)

    def get_rouge_score_for_snums(self, snums, article):
        ex = self.extract_summary(article, sentnums=snums)
        return ex.get_mean_score()

    @abstractmethod
    def _extract_sentums(self, article, **kwargs):
        """
        Returns the sentnums extracted by the model. To be overriden
        in implementing subclasses
        :param article: Article
        :return: List of sentnums extracted
        """
        return []


# base extraction
class Extraction(object):
    """
    Base class for an extractive summary. Stores all the primary information
    related to a summary, including the original article's filename, the summary
    itself, the corresponding sentence numbers, and the accuracy of the
    extractive summary held in the class.
    """
    def __init__(self, doc_fname, sent_nums, summary, results):
        super(Extraction, self).__init__()
        self.fname = doc_fname
        self.sents = sent_nums
        self.summ = summary
        self.rouge_res = results
        if type(self.rouge_res) is dict:
            self.rouge_res['mean'] = {'f': 0, 'p': 0, 'r': 0}
            for key in self.rouge_res['mean']:
                total = sum(self.rouge_res[score_type][key] for score_type in self.rouge_res)
                self.rouge_res['mean'][key] = total / 3

    def __repr__(self):
        return f'{self.sents}: {self.summ}'

    def get_mean_score(self):
        if type(self.rouge_res) == dict:
            if 'mean' in self.rouge_res:
                return self.rouge_res['mean']['r']
            else:
                raise NotImplementedError('This should never be reached.')
        return self.rouge_res

    def get_res_str(self, tabs=2):
        s = []
        for res_name, res in self.rouge_res.items():
            s.append('{}{}: {:6.4f}'.format(tabs*'\t', res_name, res['r']))
        return '\n'.join(s)

    def get_summary_str(self, tabs=2):
        return '\n'.join(map(lambda s: '{}{}'.format(tabs*'\t', s), self.summ))

    def get_snum_str(self):
        return ', '.join(map(str, self.sents))
