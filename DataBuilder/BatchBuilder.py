from collections import Counter, defaultdict
import tensorflow as tf
import os


class BatchBuilder:

    def __init__(self, basedir, batch_size=4, min_after_deque=10):
        self.basedir = basedir
        self.batch_size = batch_size
        self.min_after_dequeue = min_after_deque

    @staticmethod
    def __initialize_queue(files, randomize=True):
        """Convert images to Tensors

        Takes a list of image files with the same file format and turns them into Tensors.

        Args:
            *files (char): List of string cointaining the path and file matching pattern.
            randomize (boolean): Organize the files in the queue in a random sequence.
                Default: True

        Returns:
            Dictionary of Tensors containing the queues with all the images which match the pattern
        """
        dataset_queue = {}

        for i in range(len(files)):
            match = tf.train.match_filenames_once(files[i])
            queue = tf.train.string_input_producer(match, shuffle=randomize)
            reader = tf.WholeFileReader()
            _, image_file = reader.read(queue)
            dataset_queue.update({'dataset' + str(i): image_file})

        return dataset_queue

    def __image_to_jpeg(self, files, labels, width=80, height=140, gray=True, normalize=True,
                        method=0, randomize=False, resize=True, center=False):
        """Configure a set of pictures

        Rescale, assign labels or turn a picture into RGB/Grayscale colormap. It also
        normalizes the pictures on a Cartesian centre.

        Args:
            *files (Tensor): List of string containing a set of pictures from a directory.
            *labels (int): Integer list with a single element to label each Tensor created
                by this function.
            width (int): Image WIDTH. Default: 80
            height (int): Image height. Default: 140
            gray (boolean): True to turn a picture into grayscale. False to turn it into
                RGB colorspace. Default: True
            normalize (boolean): True to center images on the Cartesian origin. False to
                keep it as it is. Default: True
            method (int): Select a method to resize the images with a number mapping to:
                    0 -> ResizeMethod.BILINEAR
                    1 -> ResizeMethod.NEAREST_NEIGHBOR
                    2 -> ResizeMethod.BICUBIC
                    3 -> ResizeMethod.AREA
                Default: 0

        Returns:
            Dictionary containing images with the set options and a label dictionary
        """

        dataset_queue = self.__initialize_queue(files, randomize=randomize)

        dataset = {}
        labelset = {}

        for i in range(len(dataset_queue)):

            if gray:
                raw_image, label = \
                    tf.image.decode_jpeg(dataset_queue['dataset' + str(i)], channels=1), [labels[i]]
            else:
                raw_image, label = \
                    tf.image.decode_jpeg(dataset_queue['dataset' + str(i)], channels=3), [labels[i]]

            if resize:
                raw_image = tf.image.resize_images(raw_image, (width, height), method=method)

            if gray:
                raw_image = tf.reshape(raw_image, [width, height, 1])  # -1
            else:
                raw_image = tf.reshape(raw_image, [width, height, 3])  # -1

            if normalize:
                raw_image = tf.to_float(raw_image) / 256.
            else:
                raw_image = tf.to_float(raw_image)

            if center:
                raw_image = raw_image - 0.5

            dataset.update({'dataset' + str(i): raw_image})
            labelset.update({'labelset' + str(i): label})

        return dataset, labelset

    def __initialize_batches(self, dataset, labelset):
        # if training_percentage > 1 or training_percentage < 0:
        #    print("The training percentage must range between [0, 1]")
        #    print("Setting values by default . . .")
        #    training_percentage = 0.8

        capacity = self.min_after_dequeue + 3 * self.batch_size

        # Dictionary of batches
        sample_batches = {}
        label_batches = {}

        # List of batches
        batches = []
        labels = []

        for i in range(len(dataset)):
            sample_batch, label_batch = tf.train.shuffle_batch(
                [dataset['dataset' + str(i)], labelset['labelset' + str(i)]],
                batch_size=self.batch_size, capacity=capacity,
                min_after_dequeue=self.min_after_dequeue)
            sample_batches.update({'batch' + str(i): sample_batch})
            label_batches.update({'label' + str(i): label_batch})

            batches.append(sample_batches['batch' + str(i)])
            labels.append(label_batches['label' + str(i)])
            print("=" * 40)
            print()
            print("BATCH" + str(i) + " :", batches[i])
            print("LABEL" + str(i) + " :", labels[i])
            print()
        print("=" * 40)
        print()

        # Beware here so you can extract the batches for testing set
        # for i in range(len(dataset)):

        all_samples_batch = tf.concat(values=batches, axis=0)
        all_labels_batch = tf.one_hot(indices=tf.concat(values=labels, axis=0), depth=len(labels))

        return all_samples_batch, all_labels_batch

    def generate_datasets(self):
        datasets = os.listdir(self.basedir)

        all_datasets_path = {}
        num_files = defaultdict(Counter)
        path = []

        for dataset in datasets:
            classes = os.listdir(self.basedir + '/' + dataset)
            for the_class in classes:
                path.append(self.basedir + '/' + dataset + '/' + the_class + '/' + '*.jpg')
                num_files[dataset][the_class] = len(os.listdir(self.basedir + '/' + dataset + '/' + the_class))
            all_datasets_path.update({dataset + '_files': path})
            path = []

        labels = [x for x in range(len(list(all_datasets_path.values())[0]))]  # labels = [0, 1, 2, 3]

        return all_datasets_path, labels, datasets, num_files

    def get_batches(self, all_datasets_path, labels, datasets, gray=True, normalize=True, method=3,
                    randomize=False, resize=False, width=80, height=140, center=False):
        all_batches = {}

        for dataset in datasets:
            # Get training batches
            the_dataset, labelset = self.__image_to_jpeg(all_datasets_path[dataset + '_files'], labels,
                                                         gray=gray, normalize=normalize, method=method,
                                                         randomize=randomize, resize=resize, width=width,
                                                         height=height, center=center)

            dataset_batch, one_hot_label = self.__initialize_batches(the_dataset, labelset)

            all_batches.update({dataset: (tf.squeeze(dataset_batch), one_hot_label)})

        return all_batches
