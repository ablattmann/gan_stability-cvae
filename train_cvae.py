import os
import tensorflow as tf
from absl import flags
import datetime

import utils

gpu = str(utils.choose_gpu())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

from eval_model import Evaluator, eval_cvae
from cvae import CVAE

gfile = tf.gfile

train_dir = os.path.dirname(os.path.realpath(__file__))

flags.DEFINE_string('checkpoint_dir', os.path.join(train_dir, '../../data/wiki-art/cvae/checkpoints'),
                    'Directory, where the data to feed is located.')
flags.DEFINE_integer('batch_size', 32, 'The batch_size for the model.')
flags.DEFINE_integer('shuffle_buffer_size', 10000, 'Number of records to load '
                                                   'before shuffling and yielding for consumption. [100000]')
flags.DEFINE_integer('save_summaries_steps', 300, 'Number of seconds between '
                                                  'saving summary statistics. [1]')  # default 300
flags.DEFINE_integer('save_checkpoint_secs', 1200, 'Number of seconds between '
                                                   'saving checkpoints of model. [1200]')

flags.DEFINE_integer('n_steps', 3000000, 'The total number of train steps to take.')
flags.DEFINE_bool('load_model', False, 'Whether to load from existing weights or train new from scratch.')

FLAGS = flags.FLAGS


def main(_):
    print('learning_rate ', FLAGS.learning_rate)
    print('Adam\'s beta parameter ', FLAGS.optimizer_add)
    print('data_dir ', FLAGS.data_dir)
    print(FLAGS.batch_size)
    print('gf_ef_dim', FLAGS.gf_dim, FLAGS.ef_dim)
    print('Use conditional imstance normalization? ', FLAGS.use_cin)
    print('Starting the program..')
    gfile.MakeDirs(FLAGS.checkpoint_dir)

    if FLAGS.load_model:
        model_dir = 'cvae_wiki-art_32_128_2019-01-07-16-10-51'
        logdir = os.path.join(FLAGS.checkpoint_dir, model_dir)
        print('reloading existing model and continuing training.')
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_dir = '%s_%s_%s_%s' % (
            'cvae_wiki-art', FLAGS.batch_size, FLAGS.image_size, now)
        logdir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    print('checkpoint_dir: {}'.format(FLAGS.checkpoint_dir))
    print('model_dir: {}'.format(model_dir))

    gfile.MakeDirs(logdir)

    with tf.Graph().as_default():
        # Set up device to use
        device = '/gpu:0'

        with tf.device(device):
            # Instantiate global_step.
            global_step = tf.train.create_global_step()

        # create model graph
        cvae = CVAE(FLAGS, global_step, device)
        evaluator = Evaluator(cvae.dec_fcn, FLAGS, device, logdir)

        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        # train
        tf.contrib.training.train(cvae.train_op, logdir=logdir, hooks=(
            [tf.train.StopAtStepHook(num_steps=300000), evaluator.compute_stats(),
             eval_cvae(cvae.dec_fcn, device, FLAGS, logdir, FLAGS.z_g_dim)]),
                                  save_summaries_steps=FLAGS.save_summaries_steps,
                                  save_checkpoint_secs=FLAGS.save_checkpoint_secs, config=session_config)


if __name__ == '__main__':
    tf.app.run()
