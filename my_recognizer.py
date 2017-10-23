import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    sequences = list(test_set.get_all_Xlengths().values())
    guess = ''

    for X, length in sequences:
        best_log_likehood = float('inf')
        prob = {}

        for word, model in models.items():
            try:
                log_likehood = model.score(X, length)
            except:
                log_likehood = float("-inf")

            prob[word] = log_likehood

            if log_likehood > best_log_likehood:
                best_log_likehood = log_likehood
                guess = word

        probabilities.append(prob)
        guesses.append(guess)

    return probabilities, guesses