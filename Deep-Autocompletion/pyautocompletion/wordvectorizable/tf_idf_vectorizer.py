# -*- coding: utf-8 -*-
from pyautocompletion.word_vectolizerable import WordVectolizerable


class TfIdfVectorizer(WordVectorizerable):
    '''
    Tf-Idf Vectorizer.
    '''
    
    def __init__(self):
        raise NotImplementedError("This class must be implemented.")
    
    def fit(self, sentence_token_arr):
        '''
        Fit.
        
        Args:
            sentence_token_arr:    np.ndarray of tokens.
        '''
        pass

    def vectorlize(self, token_list):
        '''
        Vectorize.

        Args:
            token_list:   List of tokens.

        Returns:
            np.array([vectors])
        '''
        pass
