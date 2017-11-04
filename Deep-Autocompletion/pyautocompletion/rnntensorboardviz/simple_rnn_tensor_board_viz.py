import tensorflow as tf
import numpy as np
from rnn_tensor_board_viz import RnnTensorBoardViz


class SimpleRnnTensorBoardViz(RnnTensorBoardViz):
    '''
    RNN demo.
    '''

    def preprocess(self, x_arr, t_arr):
        '''
        Preprocess data.
        
        Args:
            x_arr:    X.
            t_arr:    Target(label) data.
        
        '''
        train_indices = np.random.choice(x_arr.shape[0], round(0.8 * x_arr.shape[0]), replace=False)
        test_indices = np.array(list(set(range(x_arr.shape[0])) - set(train_indices)))
        self.train_x_arr = x_arr[train_indices]
        self.test_x_arr = x_arr[test_indices]
        self.train_t_arr = t_arr[train_indices]
        self.test_t_arr = t_arr[test_indices]

    def session_run(self, training_num, batch_size, summary_freq):
        '''
        Execute mini-batch training and add summary.
        
        Args:
            training_num:  Number of training.
            batch_size:    batch size.
            summary_freq:  Frequency of tf.summary.FileWriter().add_summary().
        '''
        i = 0
        for _ in range(training_num):
            i += 1
            rand_index = np.random.choice(self.train_x_arr.shape[0], size=batch_size)
            batch_x = self.train_x_arr[rand_index]
            batch_t = self.train_t_arr[rand_index]
            self.sess.run(self.opt_algo, feed_dict={self.x: batch_x, self.t: batch_t})
            if i % summary_freq == 0:
                summary, loss, accuracy = self.sess.run(
                    [self.summary, self.loss, self.accuracy],
                    feed_dict={
                        self.x: self.test_x_arr,
                        self.t: self.test_t_arr
                    }
                )
                self.writer.add_summary(summary, i)
                print("Step: " + str(i) + " Loss: " + str(loss) + " Accuracy: " + str(accuracy))
