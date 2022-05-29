import tensorflow as tf


class FFNN(tf.keras.Model):
    # The constructor of the FFNN model
    def __init__(self, units1, units2, activation, random_seed, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(units1, activation=activation,
                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                   stddev=0.05,
                                                                                                   seed=random_seed))
        self.hidden2 = tf.keras.layers.Dense(units2, activation=activation,
                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                   stddev=0.05,
                                                                                                   seed=random_seed))
        self.label_output = tf.keras.layers.Dense(2, activation='sigmoid', name='label')

    # Forward function of the FFNN model
    def call(self, inputs, training=None):
        hidden1 = self.hidden1(inputs)
        hidden2 = self.hidden2(hidden1)
        label_output = self.label_output(hidden2)
        return label_output

    # This method is help to print out the summary of the model which is using keras subclassed API
    def model(self):
        x = tf.keras.Input(shape=(2400, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


# A FFNN model which contains multi-output
class FFNN_multiOutput(tf.keras.Model):
    def __init__(self, units1, units2, activation, random_seed, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(units1, activation=activation,
                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                   stddev=0.05,
                                                                                                   seed=random_seed))
        self.hidden2 = tf.keras.layers.Dense(units2, activation=activation,
                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                   stddev=0.05,
                                                                                                   seed=random_seed))
        self.domain_output = tf.keras.layers.Dense(3, activation='softmax', name='domain')
        self.label_output = tf.keras.layers.Dense(2, activation='sigmoid', name='label')

    def call(self, inputs, training=None):
        hidden1 = self.hidden1(inputs)
        hidden2 = self.hidden2(hidden1)
        domain_output = self.domain_output(hidden2)
        label_output = self.label_output(hidden2)

        return [label_output, domain_output]

    def model(self):
        x = tf.keras.Input(shape=(2400, 2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
