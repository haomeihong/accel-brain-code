# -*- coding: utf-8 -*-
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pyautocompletion.wordvectorizable.word_2_vec_vectorizer import Word2VecVectorizer
from pyautocompletion.preprocess_document import PreprocessDocument
from pyautocompletion.rnntensorboardviz.lstm_rnn_tensor_board_viz import LSTMRnnTensorBoardViz


def Main(document, dimention=5):
    '''
    Entry Point.
    '''
    nlp_base = AutoAbstractor()
    tokenizable_doc = MeCabTokenizer()
    nlp_base.tokenizable_doc = tokenizable_doc
    word_vectorizable = Word2VecVectorizer(
        path_to_corpus="corpus.txt",
        generate_corpus_flag=False,
        dimention=dimention
    )
    preprocess_document = PreprocessDocument(
        nlp_base=nlp_base,
        word_vectorizable=word_vectorizable
    )
    feature_arr, class_arr = preprocess_document.preprocess(document)
    #print(feature_arr)
    #print(class_arr)
    rnn_tensor_board_viz = LSTMRnnTensorBoardViz()
    rnn_tensor_board_viz.initialize(
        x_shape=(None, feature_arr.shape[1], dimention),
        t_shape=(None, class_arr.shape[1], dimention),
        learning_rate=0.000001,
        class_num=dimention,
        cell_units_num=3,
        log_dir="/tmp/tensorboard_lstm_rnn"
    )
    rnn_tensor_board_viz.preprocess(feature_arr, class_arr)
    rnn_tensor_board_viz.session_run(
        batch_size=3,
        training_num=2000,
        summary_freq=1
    )

if __name__ == "__main__":
    import sys
    document = sys.argv[1]

    Main(document)
