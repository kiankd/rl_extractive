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

class Summarizer(ABC):

    @abstractmethod
    def extract_summary(self, article):
        pass
