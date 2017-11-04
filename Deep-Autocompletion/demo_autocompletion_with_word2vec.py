# -*- coding: utf-8 -*-
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pyautocompletion.wordvectorizable.word_2_vec_vectorizer import Word2VecVectorizer
from pyautocompletion.preprocess_document import PreprocessDocument


def Main(document):
    '''
    Entry Point.
    '''
    nlp_base = AutoAbstractor()
    tokenizable_doc = MeCabTokenizer()
    nlp_base.tokenizable_doc = tokenizable_doc
    word_vectorizable = Word2VecVectorizer(
        path_to_corpus="corpus.txt",
        generate_corpus_flag=False,
        dimention=100
    )
    preprocess_document = PreprocessDocument(
        nlp_base=nlp_base,
        word_vectorizable=word_vectorizable
    )
    sentence_token_arr = preprocess_document.preprocess(document)
    print(sentence_token_arr)

if __name__ == "__main__":
    import sys
    document = sys.argv[1]

    Main(document)
