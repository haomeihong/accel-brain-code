# -*- coding: utf-8 -*-
from pysummarization.nlp_base import NlpBase
from pyautocompletion.word_vectorizerable import WordVectorizerable
import numpy as np


class PreprocessDocument(object):
    '''
    Preprocess for Encoder-Decoder.
    '''
    
    # The object of `NlpBase`.
    __nlp_base = None
    # The object of `WordVectolizerable`.
    __word_vectorizable = None
    
    def __init__(self, nlp_base, word_vectorizable):
        '''
        Initialize.
        
        Args:
            nlp_base:           The object of `NlpBase`.
            word_vectorizable:  The object of `WordVectorizerable`.
        '''
        if isinstance(nlp_base, NlpBase):
            self.__nlp_base = nlp_base
        else:
            raise TypeError("The type of `nlp_base` must be `NlpBase`.")

        if isinstance(word_vectorizable, WordVectorizerable):
            self.__word_vectorizable = word_vectorizable
        else:
            raise TypeError("The type of `word_vectorizable` must be `WordVectorizerable`.")

    def preprocess(self, document):
        '''
        Preprocess.
        
        Args:
            document:    document.
        
        Returns:
            np.array(
                [
                    ["token1", "token2", "token3"],
                    ["token4", "token5", "token6"]
                ]
            )
        '''
        sentence_list = self.__nlp_base.listup_sentence(document)
        sentence_token_list = []
        length_list = []
        for sentence in sentence_list:
            self.__nlp_base.tokenize(sentence)
            token_list = self.__nlp_base.token
            sentence_token_list.append(token_list)
            length_list.append(len(token_list))
        max_length = max(length_list)
        for i in range(len(sentence_token_list)):
            while len(sentence_token_list[i]) < max_length:
                sentence_token_list[i].append("<EOC>")
        sentence_token_arr = np.array(sentence_token_list)
        self.__word_vectorizable.fit(sentence_token_arr)
        def vectorize(elm_list):
            return self.__word_vectorizable.vectorize(elm_list)
        vector_arr = np.apply_along_axis(vectorize, 1, sentence_token_arr)
        return vector_arr
