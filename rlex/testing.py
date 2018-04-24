import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from rlex.abstract_extraction import Params
from rlex.rl_extraction import PolicyGradientExtractor, RESULTS
from rlex.oracle_extraction import Lead3Summarizer, GreedyOracleSummarizer, RandomSummarizer
from rlex.load_data import get_samples
from rlex.helpers import PATH_TO_RESULTS, scores_to_str

np.random.seed(1917)

def store_ex_result(score_holder, verbose=False):
    if verbose:
        print(ex)
        print('{} results:'.format(model.name))
    for key, res in ex.rouge_res.items():
        if verbose:
            print('\t{}: {}'.format(key, scores_to_str(res)))
        score_holder[model.name][key].append(res['r'])

def print_model_score_res(score_holder):
    for name, all_scores in score_holder.items():
        print('\n{} ROUGE Recall scores:'.format(name))
        for key, scores in all_scores.items():
            print('\t{}: mean = {:5.3f}, std = {:5.3f}'.format(
                key, np.mean(scores), np.std(scores),
            ))

if __name__ == '__main__':
    # get an article
    CLEAN_ARTICLES = True
    outdir = ''.join([PATH_TO_RESULTS,
                      'clean_' if CLEAN_ARTICLES else 'dirty_',
                      'extrs/'])

    models = [RandomSummarizer(1848), Lead3Summarizer(), GreedyOracleSummarizer('mean')]
    test_params = [
        Params(gamma=1, v_lr=0.05, p_lr=0.15, use_baseline=True, update_only_last=True),
    ]
    models.extend(PolicyGradientExtractor(p) for p in test_params)
    train_article_scores = {m.name: {'rouge-1': [],
                             'rouge-2': [],
                             'rouge-l': [],
                             'mean': [] } for m in models}
    new_articles_scores = deepcopy(train_article_scores)

    articles = get_samples(clean=CLEAN_ARTICLES)
    END_ART_IDX = 200
    TRAIN_ARTICLES = articles[0: END_ART_IDX]
    PLOT = True
    SINGLE_ARTICLE_TRAINING = False
    BATCH_ARTICLE_TRAINING = True
    VERBOSE = False
    BATCH_MEAN = True

    features = {'pca_features': 1000, 'tfidf_max_features': 5000}

    # test all models
    for i, model in enumerate(models):

        # set features
        if model.is_learner():
            print('Feature extraction...')
            model.set_features(articles, **features) # extract from ALL articles®

        # batch article training
        if BATCH_ARTICLE_TRAINING:
            if model.is_learner():
                print('Big batch training...')
                results = model.train_on_batch_articles(500,
                                                        articles=TRAIN_ARTICLES,
                                                        track_greedy=PLOT,
                                                        shuffle=False,
                                                        batch_mean=BATCH_MEAN)
                if PLOT:
                    tests = [RESULTS.returns, RESULTS.greedy_scores]
                    tests = [f'{key}-mean' for key in tests]
                    lines = ['b--', 'r--', 'k--', 'g-']
                    for key, line in zip(tests, lines):
                        plt.figure()
                        plt.title('{} -- batch training {}'.format(key, 'BM' if BATCH_MEAN else ''))
                        x = list(range(len(results[key])))
                        plt.plot(x, results[key], line)
                        plt.xlabel('Training episode number')
                        plt.show()

            for j, a in enumerate(TRAIN_ARTICLES):
                ex = model.extract_summary(a)
                a.add_extraction_pred(model.name, ex)
                store_ex_result(train_article_scores, VERBOSE)

        # SINGLE ARTICLE TESTING
        elif SINGLE_ARTICLE_TRAINING:
            for j, a in enumerate(TRAIN_ARTICLES):
                if model.is_learner(): # then we r doing RL, train first
                    print('Training...')
                    sents, train_res = model.train_on_article(j, 1000, store_all_changes=PLOT)
                    if PLOT:
                        tests = [RESULTS.w_pgr, RESULTS.w_vpi, RESULTS.policies]
                        tests += [RESULTS.returns, RESULTS.greedy_scores]
                        lines = ['b--', 'r--', 'k--', 'g-']
                        for key, line in zip(tests, lines):
                            values = []
                            if key == RESULTS.returns or key == RESULTS.greedy_scores:
                                values = train_res[key]
                            else:
                                w_ot = train_res[key]  # weights over time
                                for widx in range(1, len(w_ot)):
                                    values.append(np.linalg.norm(w_ot[widx] - w_ot[widx-1]))
                            plt.figure()
                            plt.title('{} -- article {}'.format(key, j))
                            x = list(range(len(values)))
                            plt.plot(x, values, line)
                            plt.xlabel('Episode number')
                            plt.show()

                        policy = train_res[RESULTS.policies][-1].reshape(-1, 11)
                        plt.figure()
                        plt.title('Last Policy')
                        plt.imshow(policy, cmap='hot')
                        plt.show()
                        print('Max probability: {:2.5f}'.format(np.max(policy)))
                        print('Min probability: {:2.5f}'.format(np.min(policy)))

                ex = model.extract_summary(a)
                a.add_extraction_pred(model.name, ex)
                store_ex_result(train_article_scores, VERBOSE)

    # print full agglomerated results

    print('\n\nRESULTS ON TRAINING ARTICLES:')
    print_model_score_res(train_article_scores)
    for a in TRAIN_ARTICLES:
        a.serialize_extr_results(PATH_TO_RESULTS + 'train_arts/')

    print('\n\nRESULTS ON VALIDATION ARTICLES:')
    for a in articles[END_ART_IDX:]:
        for model in models:
            ex = model.extract_summary(a)
            for key, scores in ex.rouge_res.items():
                store_ex_result(new_articles_scores)
            a.add_extraction_pred(model.name, ex)
        a.serialize_extr_results(PATH_TO_RESULTS + 'test_arts/')
    print_model_score_res(new_articles_scores)

