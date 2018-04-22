import numpy as np
from overrides import overrides
from rlex.feature_extraction import extract_sentence_doc_features
from rlex.abstract_extraction import NUM_SENTS_EXTRACT, Extractor

# results tracking constants
class RESULTS(object):
    states = 'states'
    actions = 'actions'
    all_sa_feats = 'all_sa_feats'
    policies = 'policies'
    w_pgr = 'w_pgr'
    w_vpi = 'w_vpi'
    greedy_scores = 'greedy_scores'
    returns = 'returns'

    @staticmethod
    def get_res_names__():
        return [s for s in RESULTS.__dict__
                if not s.startswith('__') and not s.endswith('__')]

# nice error
class FeaturesNotSetError(Exception):
    def __init__(self, msg):
        super(FeaturesNotSetError, self).__init__(msg)


# the main big boy
class PolicyGradientExtractor(Extractor):
    def __init__(self, params):
        """
        :param params: Params, namespace of parameters.
        """
        self.arts_sents_feats = None
        self.art_feats = None
        self.w_pgr = None # weights for policy gradient
        self.w_vpi = None # weights for baseline val fun of policy grad
        self.policy = None
        self.articles = None
        self.articles_to_idx = None
        super(PolicyGradientExtractor, self).__init__(params.to_name(), params)

    @staticmethod
    @overrides
    def is_learner():
        return True

    def feature_check(self):
        if self.arts_sents_feats is None:
            raise FeaturesNotSetError('You did not set article features!')

    def train_on_article(self, aidx, n_episodes, store_all_changes=False):
        self.feature_check()

        # this is for tracking results
        all_changes = None
        if store_all_changes:
            all_changes = {res: [] for res in RESULTS.get_res_names__()}
            all_changes[RESULTS.w_pgr].append(self.w_pgr.copy())
            all_changes[RESULTS.w_vpi].append(self.w_vpi.copy())

        time = 1
        for i in range(n_episodes):
            if (i + 1) % 5 == 0:
                time += 1

            # get the full trajectory
            trajectory = self.__generate_trajectory(aidx)
            states, actions, all_sa_feats, policies, rouge_score = trajectory

            # update weights and stuff
            self.__mc_learning_update(states, actions, all_sa_feats, policies,
                                      rouge_score, use_baseline=self.params.use_baseline)

            # track mad results, if desired
            if store_all_changes:
                all_changes[RESULTS.states].extend(states)
                all_changes[RESULTS.actions].extend(actions)
                all_changes[RESULTS.all_sa_feats].extend(all_sa_feats)
                all_changes[RESULTS.policies].append(policies[0])
                all_changes[RESULTS.w_pgr].append(self.w_pgr.copy())
                all_changes[RESULTS.w_vpi].append(self.w_vpi.copy())
                all_changes[RESULTS.returns].append(rouge_score)

                extr = self.extract_summary(self.articles[aidx])
                greedy_score = extr.get_mean_score()
                all_changes[RESULTS.greedy_scores].append(greedy_score)

        return all_changes

    def train_on_batch_articles(self, article_training_steps, articles=None,
                                track_greedy=False, track_results=True,
                                shuffle=False, batch_mean=True):
        # initialization check
        self.feature_check()
        if articles is None:
            articles = self.articles

        if track_results:
            results = {RESULTS.returns: [], RESULTS.greedy_scores: []}
            for key in set(results.keys()):
                results[f'{key}-mean'] = []

        # iterate over number of training steps
        art_idxs = list(range(len(articles)))
        for n in range(article_training_steps):
            if shuffle: self.params.random.shuffle(art_idxs)

            # storing our update vectors over time (if using batch-mean)
            all_pgr_updates, all_vpi_updates = [], []
            for aidx in art_idxs:
                # get the full trajectory
                trajectory = self.__generate_trajectory(aidx)
                states, actions, all_sa_feats, policies, rouge_score = trajectory

                # get weight updates
                pgr_updates, vpi_updates = self.__mc_learning_update(
                    states, actions, all_sa_feats, policies, rouge_score,
                    use_baseline=self.params.use_baseline,
                    get_updates_only=bool(batch_mean))

                # update tracking
                if bool(batch_mean):
                    all_pgr_updates += pgr_updates
                    all_vpi_updates += vpi_updates

                # store all returns
                if track_results:
                    results[RESULTS.returns].append(rouge_score)

            # store mean return
            if track_results:
                results[f'{RESULTS.returns}-mean'].append(np.mean(results[RESULTS.returns]))

            # update weights with mean update vector from batch, if using batch's mean target
            if bool(batch_mean):
                self.w_pgr += self.params.p_lr * np.sum(np.array(all_pgr_updates), axis=0) \
                                / len(all_pgr_updates)
                self.w_vpi += self.params.v_lr * np.sum(np.array(all_vpi_updates), axis=0) \
                                / len(all_vpi_updates)

            # track the results of a greedy policy - EXPENSIVE!
            if track_greedy:
                key = RESULTS.greedy_scores
                for article in articles:
                    extr = self.extract_summary(article)
                    greedy_score = extr.get_mean_score()
                    results[key].append(greedy_score)
                results[f'{key}-mean'].append(np.mean(results[key]))

        if track_results:
            return results

    def set_features(self, articles, **kwargs):
        """
        Set params and initialize weights, action-features, state-features
         , and state-action-features using the sizes of the feature-storage
         objects (the sentence features for each article), and also the
         article features as their own things for states alone.
        :param articles: List of Article - pass this to set all the model's
            features for the given article.
        :return: None
        """
        self.articles = articles
        self.articles_to_idx = {a: i for i, a in enumerate(self.articles)}
        self.arts_sents_feats, self.art_feats = extract_sentence_doc_features(articles, **kwargs)
        self.params.set_params(a_feats=self.arts_sents_feats[0].shape[1],
                               s_feats=self.art_feats[0].shape[0],)
        self.params.set_params(sa_feats=self.params.a_feats + self.params.s_feats)
        self.w_pgr = np.zeros(self.params.sa_feats)
        self.w_vpi = np.zeros(self.params.s_feats)

        # set all features to be one hots
        # if basic:
        #     new_feats = []
        #     for sent_feats in self.arts_sents_feats:
        #         new_feats.append(np.identity(len(sent_feats)))
        #     self.arts_sents_feats = new_feats

    @overrides
    def _extract_sentums(self, article, **kwargs):
        _, actions, _, _, _ = self.__generate_trajectory(self.articles_to_idx[article], greedy=True)
        return [a[0] for a in actions]

    def __generate_trajectory(self, article_idx, greedy=False):
        sents_feats = self.arts_sents_feats[article_idx]

        # initial state is the tf-idf of the document
        state = self.art_feats[article_idx]

        # generate a set of action/sentence choices
        states = [state]
        actions = []
        policies = []
        all_sa_feats = []
        nS = len(sents_feats)
        for i in range(NUM_SENTS_EXTRACT):
            _state_mat = np.stack([state for _ in range(len(sents_feats))])
            sa_features = np.concatenate((sents_feats, _state_mat), axis=1)
            policy = self.__get_policy(sa_features)
            a = self.__select_action_from_policy(policy, [a[0] for a in actions], greedy=greedy)

            # update tracking variables, action list & number of states
            all_sa_feats.append(sa_features)
            policies.append(np.copy(policy))
            actions.append((a, sa_features[a]))
            nS = nS - 1

            # update state, remove the sent's tfidf from the avg rep
            #   only do it of course if not at the last step.
            if i < NUM_SENTS_EXTRACT - 1:
                action_tfidf = sents_feats[a][:self.params.s_feats]
                state = (state - action_tfidf) * ((nS + 1) / nS)
                states.append(state)
        mc_return = self.get_rouge_score_for_snums([a[0] for a in actions], self.articles[article_idx])
        return states, actions, all_sa_feats, policies, mc_return

    def __mc_learning_update(self, states, actions, sa_feats, policies, mc_return,
                             use_baseline=False, get_updates_only=False):
        assert(len(states) == len(actions) == len(policies))
        vpi_updates = []
        pgr_updates = []
        for t in range(len(states)):
            if t != len(states) - 1 and self.params.update_only_last:
                continue

            discount = self.params.gamma ** ((NUM_SENTS_EXTRACT-1) - t)
            a, phi_sa = actions[t]
            ln_gradient = phi_sa - np.sum(policies[t].reshape(-1, 1) * sa_feats[t], axis=0)
            ln_gradient = np.array(ln_gradient).reshape(-1)
            update_target = mc_return

            # if use baseline, we are gonna learn the value function.
            if use_baseline:
                _flat_state = np.array(states[t]).flatten()
                vhat_s = np.dot(self.w_vpi, _flat_state)
                update_target = update_target - vhat_s
                vpi_updates.append(((discount * mc_return) - vhat_s) * _flat_state)

                if not get_updates_only:
                    self.w_vpi += self.params.v_lr * vpi_updates[-1]

            # now we do the actual P-G update
            pgr_updates.append(discount * update_target * ln_gradient)
            if not get_updates_only:
                self.w_pgr += self.params.p_lr * pgr_updates[-1]

        if get_updates_only:
            return pgr_updates, vpi_updates
        return None, None

    def __get_policy(self, feats):
        prefs = np.matmul(np.array(feats), self.w_pgr)
        assert(len(prefs) == len(feats))
        policy = np.e ** prefs
        return policy / np.sum(policy)

    def __select_action_from_policy(self, policy, selected_a, greedy=False):
        if greedy:
            best = np.argsort(policy)
            b_idx = -1
            while best[b_idx] in selected_a:
                b_idx -= 1
            return best[b_idx]

        elif self.params.method == 'softmax':
            r = np.random.random()
            best = np.argsort(policy)
            total = 0
            aidx = len(policy)-1
            action = best[aidx]

            # keep moving down the prob dist until we hit the random roll
            while total < r or action in selected_a:
                total += policy[aidx]
                aidx -= 1
                action = best[aidx]

            return action

        else:
            raise NotImplementedError(f'\"{self.params.method}\" action selection not implemented!')
