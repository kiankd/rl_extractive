PATH_TO_SAMPLES = '../examples_250/'
TEST_SAMPLES_PATH = '../examples_test/'
PATH_TO_RESULTS = 'results/'


def scores_to_str(score_dict):
    return {key: '{:5.3f}'.format(value) for key, value in score_dict.items()}
