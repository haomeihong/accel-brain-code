# -*- coding: utf-8 -*-
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pyautocompletion.wordvectorizable.word_2_vec_vectorizer import Word2VecVectorizer
from pyautocompletion.preprocess_document import PreprocessDocument
from pyautocompletion.rnntensorboardviz.lstm_rnn_tensor_board_viz import LSTMRnnTensorBoardViz


def Main(document, dimention=10):
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
    document = '''
自然言語処理（しぜんげんごしょり、英語: natural language processing、略称：NLP）は、人間が日常的に使っている自然言語をコンピュータに処理させる一連の技術であり、人工知能と言語学の一分野である。「計算言語学」（computational linguistics）との類似もあるが、自然言語処理は工学的な視点からの言語処理をさすのに対して、計算言語学は言語学的視点を重視する手法をさす事が多い[1]。データベース内の情報を自然言語に変換したり、自然言語の文章をより形式的な（コンピュータが理解しやすい）表現に変換するといった処理が含まれる。応用例としては予測変換、IMEなどの文字変換が挙げられる。
自然言語の理解をコンピュータにさせることは、自然言語理解とされている。自然言語理解と、自然言語処理の差は、意味を扱うか、扱わないかという説もあったが、最近は数理的な言語解析手法（統計や確率など）が広められた為、パーサ（統語解析器）などが一段と精度や速度が上がり、その意味合いは違ってきている。もともと自然言語の意味論的側面を全く無視して達成できることは非常に限られている。このため、自然言語処理には形態素解析と構文解析、文脈解析、意味解析などをSyntaxなど表層的な観点から解析をする学問であるが、自然言語理解は、意味をどのように理解するかという個々人の理解と推論部分が主な研究の課題になってきており、両者の境界は意思や意図が含まれるかどうかになってきている。
'''

    Main(document)
