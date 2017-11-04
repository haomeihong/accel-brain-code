import tensorflow as tf
from tensorflow.contrib import rnn
from abc import ABCMeta, abstractmethod
import os.path
import shutil


class RnnTensorBoardViz(metaclass=ABCMeta):
    '''
    Data Visualization for TensorBoard.
    
    Template Method Pattern:
        commonality:    Setting name scope and scalar summary.
        variability:    Setting neural network algorithms. e.g. RNN cells such as LSTM or GRU.
    
    Attributes (for Override):
        input_layer_pf:        Setting tf.placeholder of input layer.
        build_rnn_cell:        Initialize the RNN cell.
        construct_rnn:         Creates a recurrent neural network specified by `RNNCell` cell
                               for constructing Recurrent Neural Networks.
        output_layer_var:      Setting weights, biases, activation function in output layer. 
        optimizer_function:    Setting optimizer algorithm.
        evaluator_function:    Setting evaluator algorithm.

    Attributes (for Concrete method):
        x:            tf.placehodler() in input layer.
        t:            tf.placehodler() in output layer.
        p:            Activationg function in output layer.
        opt_algo:     Optimizer algorithm.
        loss:         Loss function.
        accuracy:     Evaluator algorithm (Math Reduction).
        sess:         tf.Session()
        summary:      tf.summary.merge_all()
        writer:       tf.summary.FileWriter()

    '''
    # Logging directory for TensorBoard
    # In __init__(), this directory is deleted.
    __log_dir = "/tmp/tensorboard_rnn/"

    # Naming conventions in TensorBoard's name scope.
    name_scope_dict = {
        "placeholder": "Placeholder",
        "cell": "Cell",
        "construct_rnn": "ConstructRNN",
        "output_layer": "OutputLayer",
        "optimizer": "Optimizer",
        "evaluator": "Evaluator"
    }

    # Naming conventions in TensorBoard's scalar summary.
    scalar_summary_dict = {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "weights": "Weights",
        "biases": "Biases"
    }

    def initialize(
        self,
        x_shape,
        t_shape,
        class_num,
        cell_units_num,
        learning_rate=0.001,
        log_dir=None,
        name_scope_dict=None,
        scalar_summary_dict=None
    ):
        '''
        initialize.
        
        Args:
            x_shape:               The shape of the x tensor to be fed.
            t_shape:               The shape of the tensor to be fed.
            class_num:             Number of classes(labels).
            cell_units_num:        The number of units in the RNN cell.
            learning_rare:         Learning rate.
            log_dir:               Logging directory for TensorBoard.
            name_scope_dict:       Naming conventions in TensorBoard's name scope.
            scalar_summary_dict:   Naming conventions in TensorBoard's scalar summary.
        '''
        if isinstance(x_shape, tuple) is False:
            raise TypeError("The type of x_shape must be tuple.")
        
        if isinstance(t_shape, tuple) is False:
            raise TypeError("The type of t_shape must be tuple.")
        
        if isinstance(class_num, int) is False:
            raise TypeError("The type of class_num must be int.")
        
        if isinstance(cell_units_num, int) is False:
            raise TypeError("The type of cell_units_num must be int.")
        
        if isinstance(learning_rate, float) is False:
            raise TypeError("The type of learning_rate must be float.")
            
        if isinstance(log_dir, str) is False and log_dir is not None:
            raise TypeError("The type of log_dir must be str.")
        
        if isinstance(name_scope_dict, dict) is False and name_scope_dict is not None:
            raise TypeError("The type of name_scope_dict must be dict.")
        
        if isinstance(scalar_summary_dict, dict) is False and scalar_summary_dict is not None:
            raise TypeError("The type of scalar_summary_dict must be dict.")

        if log_dir is not None and os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

        if log_dir is not None:
            self.__log_dir = log_dir
        if name_scope_dict is not None:
            self.name_scope_dict.update(name_scope_dict)
        if scalar_summary_dict is not None:
            self.scalar_summary_dict.update(scalar_summary_dict)

        with tf.Graph().as_default():
            x, t = self.input_layer_pf(x_shape, t_shape)
            cell = self.build_rnn_cell(cell_units_num)
            outputs, states = self.construct_rnn(cell, x)
            w, b, logits, p = self.output_layer_var(cell_units_num, class_num, outputs)
            loss, opt_algo = self.optimizer_function(t, logits, learning_rate)
            accuracy = self.evaluator_function(p, t)

            tf.summary.scalar(self.scalar_summary_dict["loss"], loss)
            tf.summary.scalar(self.scalar_summary_dict["accuracy"], accuracy)
            tf.summary.histogram(self.scalar_summary_dict["weights"], w)
            tf.summary.histogram(self.scalar_summary_dict["biases"], b)

            self.x = x
            self.t = t
            self.p = p
            self.opt_algo = opt_algo
            self.loss = loss
            self.accuracy = accuracy
            self.__start_session()

    def __start_session(self):
        '''
        Start TensorFlow session.
        '''
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.__log_dir, sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer

    def input_layer_pf(self, x_shape, t_shape):
        with tf.name_scope(self.name_scope_dict["placeholder"]):
            x = tf.placeholder(tf.float32, list(x_shape), name=self.name_scope_dict["placeholder"] + "__x")
            t = tf.placeholder(tf.float32, list(t_shape), name=self.name_scope_dict["placeholder"] + "__target")

        return (x, t)

    def build_rnn_cell(self, cell_units_num):
        '''
        Initialize the RNN cell.
        
        Args:
            cell_units_num:    Number of RNN cells units.
        Returns:
            RNNCell
        '''
        
        with tf.name_scope(self.name_scope_dict["cell"]):
            cell = rnn.BasicRNNCell(num_units=cell_units_num, activation=tf.nn.tanh)
        return cell

    def construct_rnn(self, cell, x):
        with tf.name_scope(self.name_scope_dict["construct_rnn"]):
            outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, time_major=False)
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
        return (outputs, states)

    def output_layer_var(self, cell_units_num, class_num, outputs):
        with tf.name_scope(self.name_scope_dict["output_layer"]):
            w = tf.Variable(
                tf.random_normal([cell_units_num, class_num], stddev=0.01),
                name=self.name_scope_dict["output_layer"] + "__weights"
            )
            b = tf.Variable(tf.zeros([class_num]), name=self.name_scope_dict["output_layer"] + "__biases")
            logits = tf.matmul(outputs[-1], w) + b
            p = tf.nn.softmax(logits, name=self.name_scope_dict["output_layer"] + "__activation")

        return (w, b, logits, p)

    def optimizer_function(self, t, logits, learning_rate):
        with tf.name_scope(self.name_scope_dict["optimizer"]):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=logits)
            loss = tf.reduce_mean(cross_entropy, name=self.name_scope_dict["optimizer"] + "__loss")
            opt_algo = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return (loss, opt_algo)

    def evaluator_function(self, p, t):
        with tf.name_scope(self.name_scope_dict["evaluator"]):
            total_error = tf.reduce_sum(tf.square(tf.subtract(t, tf.reduce_mean(t))))
            unexplained_error = tf.reduce_sum(tf.square(tf.subtract(t, p)))

        with tf.name_scope(self.name_scope_dict["evaluator"] + "__accuracy"):
            accuracy = tf.subtract(1.0, tf.divide(unexplained_error, total_error))

        return accuracy

    @abstractmethod
    def session_run(self, training_num, batch_size, summary_freq):
        '''
        Execute mini-batch training and add summary.
        
        Args:
            training_num:  Number of training.
            batch_size:    batch size.
            summary_freq:  Frequency of tf.summary.FileWriter().add_summary().
        '''
        raise NotImplementedError("This method must be implemented.")
