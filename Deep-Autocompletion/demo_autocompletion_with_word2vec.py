# -*- coding: utf-8 -*-
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.web_scraping import WebScraping
from pyautocompletion.wordvectorizable.word_2_vec_vectorizer import Word2VecVectorizer
from pyautocompletion.preprocess_document import PreprocessDocument
from pyautocompletion.rnntensorboardviz.lstm_rnn_tensor_board_viz import LSTMRnnTensorBoardViz
from pyautocompletion.rnntensorboardviz.lstmrnntensorboardviz.deep_lstm_rnn_tensor_board_viz import DeepLSTMRnnTensorBoardViz
import numpy as np

def Main(document, dimention=10, batch_size=100):
    '''
    Entry Point.
    '''
    nlp_base = AutoAbstractor()
    tokenizable_doc = MeCabTokenizer()
    nlp_base.tokenizable_doc = tokenizable_doc
    word_vectorizable = Word2VecVectorizer(
        path_to_corpus="corpus.txt",
        generate_corpus_flag=True,
        dimention=dimention
    )
    preprocess_document = PreprocessDocument(
        nlp_base=nlp_base,
        word_vectorizable=word_vectorizable
    )
    feature_arr, class_arr, feature_vector_arr, class_vector_arr = preprocess_document.preprocess(document)
    rnn_tensor_board_viz = DeepLSTMRnnTensorBoardViz()
    rnn_tensor_board_viz.lstm_layers = 3
    rnn_tensor_board_viz.initialize(
        x_shape=(None, feature_vector_arr.shape[1], feature_vector_arr.shape[2]),
        t_shape=(None, class_vector_arr.shape[1], class_vector_arr.shape[2]),
        class_num=dimention,
        learning_rate=0.0001,
        cell_units_num=10,
        log_dir="/tmp/tensorboard_lstm_rnn"
    )
    rnn_tensor_board_viz.preprocess(feature_vector_arr, class_vector_arr)
    rnn_tensor_board_viz.session_run(
        batch_size=batch_size,
        training_num=1000,
        summary_freq=batch_size
    )

    from scipy.spatial.distance import cosine
    import pandas as pd

    
    class ApplyCosine(object):
        __pred_class_vector_arr = None
        
        def __init__(self, pred_class_vector_arr):
            print(pred_class_vector_arr[0][0])
            self.__pred_class_vector_arr = pred_class_vector_arr
        
        def apply_method(self, value_arr):
            return cosine(self.__pred_class_vector_arr[0][0], value_arr)

    for _ in range(10):
        rand_index = np.random.choice(feature_vector_arr.shape[0], size=1)
        test_feature_arr = feature_arr[rand_index]
        test_feature_vector_arr = feature_vector_arr[rand_index]

        pred_class_vector_arr = rnn_tensor_board_viz.predict(test_feature_vector_arr)
        def fratten_arr(value_arr):
            return value_arr[0]
        fratten_class_vector_arr = np.apply_along_axis(fratten_arr, 1, class_vector_arr)

        def cosine_similarity(value_arr):
            return cosine(pred_class_vector_arr[0][0], value_arr)
        cosine_arr = np.apply_along_axis(cosine_similarity, 1, fratten_class_vector_arr)

        cosine_df = pd.DataFrame(cosine_arr, columns=["cosine"])
        class_df = pd.DataFrame(class_arr, columns=["class"])
        class_df = pd.concat([class_df, cosine_df], axis=1)
        class_df = class_df.dropna()
        class_df = class_df.drop_duplicates(["class"])
        class_df = class_df.sort_values(by=["cosine"], ascending=False)

        input_str = "".join([v for v in test_feature_arr[0] if v is not None])
        print(input_str)
        for i in range(5):
            print(" => " + class_df.iloc[i]["class"] + "(" + str(class_df.iloc[i]["cosine"]) + ")")

if __name__ == "__main__":
    import sys
    url = sys.argv[1]
     # Object of web scraping.
    web_scrape = WebScraping()
    # Web-scraping.
    document = web_scrape.scrape(url)
    Main(document)
