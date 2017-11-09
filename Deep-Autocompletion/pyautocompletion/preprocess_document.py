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

    def preprocess(self, document, max_length=None, repeated_padding_flag=False):
        '''
        Preprocess.
        
        Args:
            document:    document.
            max_length:  Max length of one sentence.
        
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

        if max_length is None:
            max_length = max(length_list)

        feature_list_list = []
        class_list = []
        for i in range(len(sentence_token_list)):
            while len(sentence_token_list[i]) < max_length:
                if repeated_padding_flag is False:
                    sentence_token_list[i].append(None)
                else:
                    sentence_token_list[i].extend(sentence_token_list[i])
                    if len(sentence_token_list[i]) > max_length:
                        sentence_token_list[i] = sentence_token_list[i][:max_length]
            for j in range(2, len(sentence_token_list[i])):
                if sentence_token_list[i][j] is not None:
                    feature_list = sentence_token_list[i][:j-1]
                    if repeated_padding_flag is False:
                        feature_list.extend([None] * len(sentence_token_list[i][j-1:]))
                    else:
                        feature_list.extend(feature_list * len(sentence_token_list[i][j-1:]))
                        feature_list = feature_list[:len(sentence_token_list[i])]
                    feature_list_list.append(feature_list)
                    class_list.append(sentence_token_list[i][j])

        sentence_token_arr = np.array(sentence_token_list)
        self.__word_vectorizable.fit(sentence_token_arr)
        feature_arr = np.array(feature_list_list)
        class_arr = np.array(class_list)
        class_arr = class_arr.reshape(-1, 1)
        def vectorize(elm_list):
            return self.__word_vectorizable.vectorize(elm_list)
        feature_vector_arr = np.apply_along_axis(vectorize, 1, feature_arr)
        class_vector_arr = np.apply_along_axis(vectorize, 1, class_arr)
        return (feature_arr, class_arr, feature_vector_arr, class_vector_arr)
