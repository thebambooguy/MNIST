from abc import ABC, abstractmethod

import tensorflow as tf


class AbstractFetch(ABC):
    def __init__(self, network, placeholders, loss='softmax', optimizer='sgd'):
        self.network = network

        self.X = placeholders.x
        self.labels = placeholders.y
        self.learning_rate = placeholders.learning_rate
        self.is_training = placeholders.is_training

        if loss == 'softmax':
            self._cost_function = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.network)
        elif loss == 'hinge':
            self._cost_function = tf.losses.hinge_loss(labels=self.labels, logits=self.network)
        elif loss == 'mse':
            self._cost_function = tf.reduce_mean(tf.square(self.X - self.network))
        elif loss == 'bce':
            self._cost_function = tf.keras.backend.binary_crossentropy(target=self.X, output=self.network)
        else:
            raise Exception(f"Bad argument: loss = '{loss}'")

        if optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif optimizer == 'Adam':
            opt = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer == 'Adagrad':
            opt = tf.train.AdagradOptimizer(self.learning_rate)
        else:
            raise Exception(f"Bad argument: optimizer = '{optimizer}'")

        self._train_step = opt.minimize(self.get_cost)

        with tf.name_scope('errors_and_accuracy'):
            self.error, self.error_op = tf.metrics.mean(self.get_cost)
            self.acc, self.acc_op = \
                tf.metrics.accuracy(labels=tf.argmax(placeholders.y, 1), predictions=tf.argmax(network, 1))

    @property
    def get_cost(self):
        return self._cost_function

    @property
    def update_weights(self):
        return self._train_step

    @abstractmethod
    def update_metrics(self):
        pass

    @abstractmethod
    def update_metrics_and_weights(self):
        pass

    @abstractmethod
    def get_metrics(self):
        pass


class Fetch(AbstractFetch):
    def update_metrics(self):
        return [self.acc_op, self.error_op]

    def update_metrics_and_weights(self):
        return [self.acc_op, self.error_op, self.update_weights]

    def get_metrics(self):
        return [self.error, self.acc]


class AutoencoderFetch(AbstractFetch):
    def update_metrics(self):
        return self.error_op

    def update_metrics_and_weights(self):
        return [self.error_op, self.update_weights]

    def get_metrics(self):
        return self.error
