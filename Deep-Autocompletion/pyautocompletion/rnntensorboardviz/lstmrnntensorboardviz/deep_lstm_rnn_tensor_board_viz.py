import tensorflow as tf
import numpy as np
from pyautocompletion.rnntensorboardviz.lstm_rnn_tensor_board_viz import LSTMRnnTensorBoardViz
from tensorflow.contrib import rnn


class DeepLSTMRnnTensorBoardViz(LSTMRnnTensorBoardViz):
    '''
    Deep LSTM RNN.
    '''
    
    # Number of layers.
    __lstm_layers = 10
    
    def get_lstm_layers(self):
        return self.__lstm_layers

    def set_lstm_layers(self, value):
        if isinstance(value, int):
            self.__lstm_layers = value
        else:
            raise TypeError()

    lstm_layers = property(get_lstm_layers, set_lstm_layers)

    def build_rnn_cell(self, cell_units_num):
        '''
        Initialize the RNN cell.
        
        Override.
        
        Args:
            cell_units_num:    Number of RNN cells units.
        Returns:
            RNNCell
        '''
        cell = super().build_rnn_cell(cell_units_num)
        if self.lstm_layers > 1:
            with tf.name_scope(self.name_scope_dict["cell"] + "__deeper"):
                cell = tf.contrib.rnn.MultiRNNCell(
                    [cell for _ in range(self.lstm_layers)]
                )
        return cell
