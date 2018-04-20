import numpy as np
from matplotlib import pyplot as plt
from rlex.abstract_extraction import Params
from rlex.rl_extraction import PolicyGradientExtractor, RESULTS
from rlex.oracle_extraction import Lead3Summarizer, GreedyOracleSummarizer
from rlex.load_data import get_samples
from rlex.helpers import PATH_TO_RESULTS, scores_to_str

np.random.seed(1917)
if __name__ == '__main__':
    # get an article
    CLEAN_ARTICLES = True
    outdir = ''.join([PATH_TO_RESULTS,
                      'clean_' if CLEAN_ARTICLES else 'dirty_',
                      'extrs/'])

    models = [Lead3Summarizer(), GreedyOracleSummarizer('mean')]
    test_params = [
        Params(v_lr=0.05, p_lr=0.15, use_baseline=True, update_only_last=True),
        # Params(v_lr=0.25, p_lr=0.55, use_baseline=True, update_only_last=True),
        # Params(v_lr=0.25, p_lr=0.60, use_baseline=True, update_only_last=True),
        # Params(v_lr=0.25, p_lr=0.65, use_baseline=True, update_only_last=True),
        # Params(v_lr=0.25, p_lr=0.70, use_baseline=True, update_only_last=True),
    ]
    models.extend(PolicyGradientExtractor(p) for p in test_params)
    model_scores = {m.name: {'rouge-1': [],
                             'rouge-2': [],
                             'rouge-l': [],
                             'mean': [],    } for m in models}

    articles = get_samples(clean=CLEAN_ARTICLES)
    TEST_ARTICLES = articles[0:100]
    PLOT = False

    for i, model in enumerate(models):
        if model.is_learner():
            print('Feature extraction...')
            model.set_features(TEST_ARTICLES, pca_features=100, tfidf_max_features=500)

        for j, a in enumerate(TEST_ARTICLES):
            if model.is_learner(): # then we r doing RL, train first
                print('Training...')
                # TODO: distributed training across a set of articles.
                sents, train_res = model.train_on_article(j, 1000,
                                      store_all_changes=PLOT, verbose=True)
                if PLOT:
                    # tests = [RESULTS.w_pgr, RESULTS.w_vpi, RESULTS.policies]
                    tests = [RESULTS.returns, RESULTS.greedy_scores]
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

            ex = model.extract_summary(a, aidx=j)
            print(ex)
            print('{} results:'.format(model.name))
            for key, scores in ex.rouge_res.items():
                print('\t{}: {}'.format(key, scores_to_str(scores)))
                model_scores[model.name][key].append(scores['r'])
            a.add_extraction_pred(model.name, ex)
        models[i] = None
    #
    # for i, a in enumerate(articles):
    #     print(a.path)

    #     a.serialize_extr_results(outdir)
    #     print()
    #     if i >= 9:
    #         break
    #
    for name, all_scores in model_scores.items():
        print('\n{} ROUGE Recall scores:'.format(name))
        for key, scores in all_scores.items():
            print('\t{}: mean = {:5.3f}, std = {:5.3f}'.format(
                key, np.mean(scores), np.std(scores),
            ))
