# -*- coding: utf-8 -*-
from pysummarization.nlp_base import NlpBase
from word_vectolizerable import WordVectolizerable
import numpy as np


class PreprocessDocument(object):
    '''
    Preprocess for Encoder-Decoder.
    '''
    
    # The object of `NlpBase`.
    __nlp_base = None
    # The object of `WordVectolizerable`.
    __word_vectorlizable = None
    
    def __init__(self, nlp_base, word_vectorlizable):
        '''
        Initialize.
        
        Args:
            nlp_base:           The object of `NlpBase`.
            word_vectorlizable: The object of `WordVectolizerable`.
        '''
        if isinstance(nlp_base, NlpBase):
            self.__nlp_base = nlp_base
        else:
            raise TypeError("The type of `nlp_base` must be `NlpBase`.")

        if isinstance(word_vectorlizable, WordVectolizerable):
            self.__word_vectorlizable = word_vectorlizable
        else:
            raise TypeError("The type of `word_vectorlizable` must be `WordVectolizerable`.")

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
        self.__vectorlizable.fit(sentence_token_arr)
        def vectorlize(elm_list):
            return self.__vectorlizable.vectorlize(elm_list)
        vector_arr = np.apply_along_axis(vectorlize, 1, sentence_token_arr)
        return vector_arr

if __name__ == "__main__":
    from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
    import sys
    nlp_base = AutoAbstractor()
    preprocess_document = PreprocessDocument(nlp_base=nlp_base)
    document = sys.argv[1]
    print(document)
    sentence_token_arr = preprocess_document.preprocess(document)
    print(sentence_token_arr)
