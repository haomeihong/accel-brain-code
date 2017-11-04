# -*- coding: utf-8 -*-
from word_vectolizerable import WordVectolizerable
from gensim.models import word2vec


class Word2VecVectorizer(WordVectolizerable):
    '''
    Word2Vec Vectorizer.
    '''
    
    # Path to corpus.
    __path_to_corpus = None
    # Dynamically generate the corpus or not.
    __generate_corpus_flag = False
    # The dimention of vectors.
    __dimention = 1000
    # Model.
    __model = None
    
    def __init__(
        self,
        path_to_corpus,
        generate_corpus_flag=False,
        dimention=1000
    ):
        '''
        Initialize.
        
        Args:
            path_to_corpus:         Path to corpus.
            generate_corpus_flag:   If `True`, this class generates the corpus dynamically.
            dimention:              The dimention of vectors.

        '''
        if isinstance(path_to_corpus, str):
            self.__path_to_corpus = path_to_corpus
        else:
            raise TypeError("The type of `path_to_corpus` must be str.")

        if isinstance(generate_corpus_flag, bool):
            self.__generate_corpus_flag = generate_corpus_flag
        else:
            raise TypeError("The type of `generate_corpus_flag` must be bool.")
        
        if isinstance(dimention, int):
            self.__dimention = dimention
        else:
            raise TypeError("The type of `dimention` must be int.")

    def fit(self, sentence_token_arr):
        '''
        Fit.
        
        Args:
            sentence_token_arr:    np.ndarray of tokens.
        '''
        if self.__generate_corpus_flag is True:
            pass

        data = word2vec.Text8Corpus(self.__path_to_corpus)
        self.__model = word2vec.Word2Vec(data, size=self.__dimention)

    def vectorlize(self, token_list):
        '''
        Vectorize.

        Args:
            token_list:   List of tokens.

        Returns:
            np.array([vectors])
        '''
        def model(v):
            if v in self.__model:
                return self.__model[v]
            else:
                return None
        return np.array([model[v] for v in token_list])
