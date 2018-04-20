import numpy as np
import random
from itertools import combinations
from progress.bar import ShadyBar
from overrides import overrides
from rlex.load_data import get_samples
from rlex.helpers import scores_to_str, PATH_TO_RESULTS
from rlex.abstract_extraction import get_score, NUM_SENTS_EXTRACT, Extractor, Params

# deterministic summarizers
class ExhaustiveOracleSummarizer(Extractor):
    def __init__(self, opt_option='mean'):
        self.opt_option = opt_option
        super(ExhaustiveOracleSummarizer, self).__init__(
            f'ExhaustiveOracle-{opt_option}',
            params=Params(opt_option=opt_option),
        )

    @overrides
    def _extract_sentums(self, article, **kwargs):
        ref = [article.get_summary_string()]
        doc_strings = []
        bad_idxs = []
        for i in range(len(article)):
            doc_strings.append(article.get_doc_sent_string(i))
            if len(article.doc_sents[i]) < 4: # 4 word sents minimum
                bad_idxs.append(i)

        best_sents = []
        best_score = -1
        combos = list(combinations(range(len(article)), NUM_SENTS_EXTRACT))
        bar = ShadyBar('Exhaustive', max=len(combos))
        for sent_idxs in combos:
            # if our sent_idxs to test contains a bad idx, skip it
            if list(filter(lambda idx: idx in bad_idxs, sent_idxs)):
                continue
            # test the Extraction
            extr = [doc_strings[idx] for idx in sent_idxs]
            score = get_score([' '.join(extr)], ref, option=self.opt_option)
            if score > best_score:
                best_sents = sent_idxs
                best_score = score
            bar.next()
        bar.finish()
        return list(sorted(best_sents))


class GreedyOracleSummarizer(Extractor):
    def __init__(self, opt_option='mean'):
        self.opt_option = opt_option
        super(GreedyOracleSummarizer, self).__init__(
            f'GreedyOracle-{opt_option}',
            params=Params(opt_option=opt_option),
        )

    @overrides
    def _extract_sentums(self, article, **kwargs):
        ref = [article.get_summary_string()]
        extr_sent_idxs = []
        for k in range(NUM_SENTS_EXTRACT):
            crt_best_idx = -1
            crt_best_score = 0
            for i in range(len(article)):
                if i in extr_sent_idxs:
                    continue
                # Here we are assuming the extracted sentences must be sorted.
                test_sent_idxs = sorted([i] + extr_sent_idxs)
                summ = [article.get_doc_sent_string(j) for j in test_sent_idxs]
                score = get_score([' '.join(summ)], ref, option=self.opt_option)
                if score >= crt_best_score:
                    crt_best_idx = i
                    crt_best_score = score
            extr_sent_idxs.append(crt_best_idx)
        return list(sorted(extr_sent_idxs))


# basic returns first 3 sents
class Lead3Summarizer(Extractor):
    def __init__(self):
        super(Lead3Summarizer, self).__init__('Lead-3', Params())

    @overrides
    def _extract_sentums(self, article, **kwargs):
        return [0, 1, 2]


# random number generator to compare with
class RandomSummarizer(Extractor):
    def __init__(self, seed=1917):
        super(RandomSummarizer, self).__init__(f'Random_seed-{seed}', Params())
        self.random = random.Random(seed)

    def _extract_sentums(self, article, **kwargs):
        sentnums = set()
        while len(sentnums) < NUM_SENTS_EXTRACT:
            sentnums.add(self.random.randint(0, len(article)-1))
        return sentnums


if __name__ == '__main__':
    CLEAN_ARTICLES = False
    articles = get_samples(clean=CLEAN_ARTICLES)
    outdir = ''.join([PATH_TO_RESULTS,
                      'clean_' if CLEAN_ARTICLES else 'dirty_',
                      'extrs/'])
    models = [
        Lead3Summarizer(),
        RandomSummarizer(1848),
        GreedyOracleSummarizer('mean'),
        GreedyOracleSummarizer('rouge-1'),
        GreedyOracleSummarizer('rouge-2'),
        GreedyOracleSummarizer('rouge-l'),
        ExhaustiveOracleSummarizer('mean'),
    ]
    model_scores = {m.name: {'rouge-1': [],
                             'rouge-2': [],
                             'rouge-l': [],} for m in models}

    for j, a in enumerate(articles):
        print(a.path)
        for model in models:
            ex = model.extract_summary(a)
            print('{} results:'.format(model.name))
            for key, scores in ex.rouge_res.items():
                print('\t{}: {}'.format(key, scores_to_str(scores)))
                model_scores[model.name][key].append(scores['r'])
            a.add_extraction_pred(model.name, ex)
        a.serialize_extr_results(outdir)
        print()
        if j >= 9:
            break

    for name, all_scores in model_scores.items():
        print('\n{} ROUGE Recall scores:'.format(name))
        for key, scores in all_scores.items():
            print('\t{}: mean = {:5.3f}, std = {:5.3f}'.format(
                key, np.mean(scores), np.std(scores),
            ))
