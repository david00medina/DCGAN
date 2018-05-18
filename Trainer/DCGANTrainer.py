import tensorflow as tf
import numpy as np
from Trainer.Trainer import Trainer
from Model.DCGAN import DCGAN
import matplotlib.pyplot as plt


class DCGANTrainer(Trainer):

    def __init__(self, image_dim, latent_dim, noise_dim, batches, epoch=10000,
                 lr_gen=0.02, lr_dis=0.02, beta1=0.5, beta2=0.999):
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.batches = batches
        self.epoch = epoch
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.beta1 = beta1
        self.beta2 = beta2
        self.outputs = None
        self.train_gen = None
        self.train_disc = None
        self.noise_input = None
        self.real_image_input = None
        self.is_training = None
        self.loss = None

    def __networks_builder(self):
        generator = DCGAN.generator(input=self.noise_input, is_training=self.is_training,
                                    latent_dim=self.latent_dim)

        discriminator_real = DCGAN.discriminator(input=self.real_image_input, is_training=self.is_training,
                                                 latent_dim=self.latent_dim)

        discriminator_fake = DCGAN.discriminator(input=generator, is_training=self.is_training,
                                                 latent_dim=self.latent_dim, reuse=True)

        stacked_gan = DCGAN.discriminator(input=generator, is_training=self.is_training,
                                          latent_dim=self.latent_dim, reuse=True)

        return generator, discriminator_real, discriminator_fake, stacked_gan

    def __loss_builder(self):
        # real images = 1, fake images = 0
        disc_loss_real = self.__get_loss(y=self.outputs[1], labels=tf.ones([self.image_dim[2]], dtype=tf.int32))
        disc_loss_fake = self.__get_loss(y=self.outputs[2], labels=tf.zeros([self.image_dim[2]], dtype=tf.int32))
        gen_loss = self.__get_loss(y=self.outputs[3], labels=tf.ones([self.image_dim[2]], dtype=tf.int32))
        return disc_loss_real, disc_loss_fake, gen_loss

    def __get_loss(self, y, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=labels))

    def __optimizer_builder(self):
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.lr_gen, beta1=self.beta1, beta2=self.beta2)
        optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.lr_dis, beta1=self.beta1, beta2=self.beta2)
        return optimizer_gen, optimizer_dis

    def load_training(self):
        # Create variables
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, self.image_dim[0], self.image_dim[1], 1])
        self.is_training = tf.placeholder(tf.bool)

        self.outputs = self.__networks_builder()

        self.loss = self.__loss_builder()

        disc_loss = self.loss[0] + self.loss[1]

        optimizer = self.__optimizer_builder()

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')

        with tf.control_dependencies(gen_update_ops):
            self.train_gen = optimizer[0].minimize(self.loss[2], var_list=gen_vars)
        disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
        with tf.control_dependencies(disc_update_ops):
            self.train_disc = optimizer[1].minimize(disc_loss, var_list=disc_vars)

        return None

    def do_training(self, sess, saver):
        for i in range(1, self.epoch + 1):
            """hola = sess.run(self.batches['training'][0])
            print(hola[0])
            plot = plt.imshow(hola[0])
            plt.show()
            print(i)"""
            batch_x, _ = self.batches.train.next_batch(self.image_dim[2])
            batch_x = np.reshape(batch_x, newshape=[-1, self.image_dim[0], self.image_dim[1], 1])
            batch_x = batch_x * 2. - 1.

            # Feed Discriminator
            z = np.random.uniform(-1., 1., size=[self.image_dim[2], self.noise_dim])
            _, dl = sess.run([self.train_disc, self.loss[0]],
                             feed_dict={self.real_image_input: batch_x,
                                        self.noise_input: z, self.is_training: True})

            # Feed Generator
            z = np.random.uniform(-1., 1., size=[self.image_dim[2], self.noise_dim])
            _, gl = sess.run([self.train_gen, self.loss[2]],
                             feed_dict={self.noise_input: z, self.is_training: True})

            if i % 500 == 0 or i == 1:
                print("EPOCH : %i || Generator Loss : %f || Descriminator Loss : %f" % (i, gl, dl))

        n = 6
        canvas = np.empty((self.image_dim[0] * n, self.image_dim[1] * n))
        for i in range(n):
            z = np.random.uniform(-1., 1., size=[n, self.noise_dim])
            g = sess.run(self.outputs[0], feed_dict={self.noise_input: z, self.is_training: False})
            g = (g + 1.) / 2.
            g = -1 * (g - 1)
            for j in range(n):
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])
        plt.figure(figsize=(n, n))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()