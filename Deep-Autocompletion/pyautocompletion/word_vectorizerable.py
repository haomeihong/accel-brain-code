# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class WordVectorizerable(metaclass=ABCMeta):
    '''
    Vectorize words.
    '''
    
    @abstractmethod
    def fit(self, sentence_token_arr):
        '''
        Fit.
        
        Args:
            sentence_token_arr:    np.ndarray of tokens.
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def vectorize(self, token_list):
        '''
        Vectorize.

        Args:
            token_list:   List of tokens.

        Returns:
            np.array([vectors])
        '''
        raise NotImplementedError("This method must be implemented.")
