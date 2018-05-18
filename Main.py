import tensorflow as tf

from BatchBuilder import BatchBuilder
from Trainer.DCGANTrainer import DCGANTrainer

# Hyperparameters
EPOCH = 10000  # 200
MIN_TRAIN_ITER = 30000
LEARNING_RATE_GEN = 0.01
LEARNING_RATE_DIS = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
UNITS = 100
DROPOUT = 0.3

# Global settings
BATCH_SIZE = 20
CLASSES = ['A', 'B', 'C', 'D']


"""filenames0 = tf.train.match_filenames_once("data/training/CAT_00/*.jpg")
filename_queue0 = tf.train.string_input_producer(filenames0, shuffle=False)
reader0 = tf.WholeFileReader()
key0, file_image0 = reader0.read(filename_queue0)
image0, label0 = tf.image.decode_jpeg(file_image0, channels=1), [0.]  # key0
image0 = tf.image.resize_images(image0, (80, 140), method=3)
image0 = tf.reshape(image0, [80, 140, 1])
image0 = tf.to_float(image0) / 256. - 0.5
batch_size = 4
min_after_dequeue = 10  # 10000
capacity = min_after_dequeue + 3 * batch_size

example_batch0, label_batch0 = tf.train.shuffle_batch([image0, label0], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
example_batch = tf.concat(values=[example_batch0], axis=0)
label_batch = tf.concat(values=[label_batch0], axis=0)
label_batch = tf.one_hot(indices=tf.to_int32(label_batch), depth=1)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    hola = sess.run(tf.squeeze(example_batch))
    print(hola[0])
    plot = plt.imshow(hola[0])
    plt.show()
"""
DATASETS_DIR = 'data'
WIDTH = 500
HEIGHT = 375

mydata = BatchBuilder(DATASETS_DIR)
all_datasets_path, labels, datasets, num_files = mydata.generate_datasets()
all_batches = mydata.get_batches(all_datasets_path, labels, datasets, gray=True,
                                 width=WIDTH, height=HEIGHT, center=False)

IMAGE_DIM = (WIDTH, HEIGHT, BATCH_SIZE)
LATENT_DIM = (7, 7, BATCH_SIZE)
NOISE_DIM = 100

trainer = DCGANTrainer(IMAGE_DIM, LATENT_DIM, NOISE_DIM, all_batches,
                       EPOCH, LEARNING_RATE_GEN, LEARNING_RATE_DIS, BETA1, BETA2)

trainer.load_training()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    trainer.do_training(sess, saver)

    #hola = sess.run(tf.squeeze(all_batches['training'][0]))
    #print(hola[0].shape)
    #plot = plt.imshow(hola[0])
    #plt.show()

"""mydata = bb.BatchBuilder(DATASETS_DIR)
all_datasets_path, labels, datasets, num_files = mydata.generate_datasets()
all_batches = mydata.get_batches(all_datasets_path, labels, datasets)"""
"""all_batches = get_batches(all_datasets_path, labels, datasets,
                          gray=True, normalize=True, method=3,
                          randomize=False, resize=True, WIDTH=28, height=28)
"""
