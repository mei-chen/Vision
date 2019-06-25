import tensorflow as tf
import model
import datetime
import numpy as np
from multiprocessing import Pool


def detect(filename):
    tfmodel_filename = "checkpoints/syde677_cnn_2017-02-14 18_39_17.432943_100832.ckpt"

    image_size = 384
    image_width = image_size
    image_height = image_size
    image_depth = 3
    output_size = 5

    with tf.Graph().as_default():
        # setup
        is_train = tf.placeholder(tf.float32)

        # im, lab, _, _, _, _, _ = kaggle_eye_image_reader.read_and_decode_imnum_pnum(
        #     filename_queue, image_size)

        # im_batch, label_batch = tf.train.batch(
        #             [im, lab],
        #             batch_size=1)

        x = tf.placeholder(tf.float32,
                           shape=[None, image_width,
                                  image_height, image_depth],
                           name='x-input')
        y_gt = tf.placeholder(tf.float32,
                              shape=[None, output_size],
                              name='y-input')

        keep_prob = tf.placeholder(tf.float32)

        y_cnn, y_cnn_softmax = model.cnn_model_9_384(x, keep_prob)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # initiate the model and training process
            print("Initiating...")
            saver.restore(sess, tfmodel_filename)
            print("Model Restored")

            image = tf.image.decode_png(open(filename, 'rb').read(), 3)
            image = tf.image.resize_images(image, [384, 384])
            image = tf.expand_dims(image, 0) / 255.0 - 0.5
            image = image.eval()

            testing_start_time = datetime.datetime.now()
            cnn_out = sess.run(
                y_cnn_softmax, feed_dict={x: image, keep_prob: 1.0, is_train: 0})

            print("Total Run Duration: ",
                  (datetime.datetime.now() - testing_start_time))

            return cnn_out


p = Pool(1)


def run(filename):
    res = p.apply(detect, (filename, ))

    idx = np.argmax(res[0])

    return {
        'label': int(idx),
        'prob': float(res[0][idx])
    }


if __name__ == '__main__':
    print(run('test.png'))
