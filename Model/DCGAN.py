import tensorflow as tf


class DCGAN:

    @staticmethod
    def generator(input, is_training, latent_dim=(7, 7, 128), reuse=False, gray=True):
        with tf.variable_scope('Generator', reuse=reuse):
            if gray:
                fc = tf.layers.dense(inputs=input,
                                     units=latent_dim[0] * latent_dim[1] * latent_dim[2], name='fc')
                norm_1 = tf.layers.batch_normalization(inputs=fc, training=is_training, name='norm_1')
                relu = tf.reshape(tf.nn.relu(norm_1, name='relu'),
                                  shape=[-1, latent_dim[0], latent_dim[1], latent_dim[2]])
                dconv_1 = tf.layers.conv2d_transpose(inputs=relu, filters=64, kernel_size=5,
                                                   strides=2, padding='same', name='dconv_1')
                norm_2 = tf.layers.batch_normalization(inputs=dconv_1, training=is_training, name='norm_2')
                relu_2 = tf.nn.relu(norm_2, name='relu_2')
                dconv_2 = tf.layers.conv2d_transpose(inputs=relu_2, filters=1, kernel_size=5,
                                                     strides=2, padding='same', name='dconv_2')
                y = tf.nn.tanh(dconv_2, name='generator_2D')
                return y
            else:
                fc = tf.layers.dense(inputs=input,
                                     units=latent_dim[0] * latent_dim[1] * latent_dim[2] * latent_dim[3],
                                     name='fc')
                norm_1 = tf.layers.batch_normalization(inputs=fc, training=is_training, name='norm_1')
                relu = tf.reshape(tf.nn.relu(norm_1, name='relu'),
                                  shape=[-1, latent_dim[0], latent_dim[1], latent_dim[2], latent_dim[3]])
                dconv_1 = tf.layers.conv3d_transpose(inputs=relu, filters=64, kernel_size=5,
                                                     strides=2, padding='same', name='dconv_1')
                norm_2 = tf.layers.batch_normalization(inputs=dconv_1, training=is_training, name='norm_2')
                relu_2 = tf.nn.relu(norm_2, name='relu_2')
                dconv_2 = tf.layers.conv3d_transpose(inputs=relu_2, filters=1, kernel_size=5,
                                                     strides=2, padding='same', name='dconv_2')
                y = tf.nn.tanh(dconv_2, name='generator_3D')

            return y

    @staticmethod
    def discriminator(input, is_training, latent_dim=(7, 7, 128), reuse=False, gray=True):
        with tf.variable_scope('Discriminator', reuse=reuse):
            if gray:
                conv_1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=5,
                                          strides=2, padding='same', name='conv_1')
                norm_1 = tf.layers.batch_normalization(inputs=conv_1, training=is_training, name='norm_1')
                lrelu_1 = tf.nn.leaky_relu(norm_1, name='lrelu_1')
                conv_2 = tf.layers.conv2d(inputs=lrelu_1, filters=128, kernel_size=5,
                                          strides=2, padding='same', name='conv_2')
                norm_2 = tf.layers.batch_normalization(inputs=conv_2, training=is_training, name='norm_2')
                lrelu_2 = tf.reshape(tf.nn.leaky_relu(norm_2, name='lrelu_2'),
                                     shape=[-1, latent_dim[0] * latent_dim[1] * latent_dim[2]])
                fc = tf.layers.dense(lrelu_2, units=1024)
                norm_3 = tf.layers.batch_normalization(inputs=fc, training=is_training, name='norm_3')
                lrelu_3 = tf.nn.leaky_relu(norm_3, name='lrelu_2')
                y = tf.layers.dense(inputs=lrelu_3, units=2, name='discriminator_2D') # Real/Fake image
            else:
                conv_1 = tf.layers.conv3d(inputs=input, filters=64, kernel_size=5,
                                          strides=2, padding='same', name='conv_1')
                norm_1 = tf.layers.batch_normalization(inputs=conv_1, training=is_training, name='norm_1')
                lrelu_1 = tf.nn.leaky_relu(norm_1, name='lrelu_1')
                conv_2 = tf.layers.conv3d(inputs=lrelu_1, filters=128, kernel_size=5,
                                          strides=2, padding='same', name='conv_2')
                norm_2 = tf.layers.batch_normalization(inputs=conv_2, training=is_training, name='norm_2')
                lrelu_2 = tf.reshape(tf.nn.leaky_relu(norm_2, name='lrelu_2'),
                                     shape=[-1, latent_dim[0] * latent_dim[1] * latent_dim[2]])
                fc = tf.layers.dense(lrelu_2, units=1024)
                norm_3 = tf.layers.batch_normalization(inputs=fc, training=is_training, name='norm_3')
                lrelu_3 = tf.nn.leaky_relu(norm_3, name='lrelu_2')
                y = tf.layers.dense(inputs=lrelu_3, units=2, name='discriminator_3D')  # Real/Fake image
            return y

    @staticmethod
    def leakyrelu(x, alpha=0.2):
        return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)