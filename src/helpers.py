PATH_TO_SAMPLES = '/Users/kian/rl_final_proj/examples/'
PATH_TO_RESULTS = '/Users/kian/rl_final_proj/src/results/'

def scores_to_str(score_dict):
    return {key: '{:5.3f}'.format(value) for key, value in score_dict.items()}
