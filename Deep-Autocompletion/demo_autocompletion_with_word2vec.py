# -*- coding: utf-8 -*-
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.web_scraping import WebScraping
from pyautocompletion.wordvectorizable.word_2_vec_vectorizer import Word2VecVectorizer
from pyautocompletion.preprocess_document import PreprocessDocument
from pyautocompletion.rnntensorboardviz.lstm_rnn_tensor_board_viz import LSTMRnnTensorBoardViz
from pyautocompletion.rnntensorboardviz.lstmrnntensorboardviz.deep_lstm_rnn_tensor_board_viz import DeepLSTMRnnTensorBoardViz
import numpy as np

def Main(document, dimention=10, batch_size=50):
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
        training_num=100,
        summary_freq=batch_size
    )
    rand_index = np.random.choice(feature_vector_arr.shape[0], size=1)
    test_feature_arr = feature_arr[rand_index]
    test_feature_vector_arr = feature_vector_arr[rand_index]

    pred_class_vector_arr = rnn_tensor_board_viz.predict(test_feature_vector_arr)
    print(pred_class_vector_arr.shape)
    print(class_vector_arr.shape)
    from scipy.spatial.distance import cosine
    import pandas as pd

    def apply_cosine(value_arr):
        print(value_arr)
        print(pred_class_vector_arr[0][0])
        return cosine(pred_class_vector_arr[0][0], value_arr)

    cosine_arr = np.apply_along_axis(apply_cosine, 1, np.transpose(class_vector_arr, (2, 0, 1)))
    print(cosine_arr)
    cosine_df = pd.DataFrame(cosine_arr, columns=["similary"])
    test_class_df = pd.DataFrame(class_arr, columns=["class"])
    test_class_df = pd.concat([test_class_df, cosine_df], axis=1)
    test_class_df = test_class_df.sort_values(by=["similary"], ascending=False)
    max_sim_class = test_class_df.iloc[0]["class"]
    test_feature_list = test_feature_arr.tolist()

    print(" ".join(test_feature_list))
    print(" => " + str(max_sim_class))


if __name__ == "__main__":
    import sys
    url = sys.argv[1]
     # Object of web scraping.
    web_scrape = WebScraping()
    # Web-scraping.
    document = web_scrape.scrape(url)
    Main(document)
