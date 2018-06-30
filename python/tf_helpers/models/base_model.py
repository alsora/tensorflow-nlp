import os
import tensorflow as tf
import data_helpers.load as load_utils
from logger_utils import get_logger

class BaseModel(object):
    """Generic base class for neural network models"""

    def __init__(self, FLAGS):

        self.FLAGS = FLAGS
        self.logger = get_logger(os.path.join(self.FLAGS.output_dir, "log.txt"))
        self.session = None
        self.saver = None



    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS.num_checkpoints)


    def restore_session(self, dir_model):
        """Reload weights into session
        Args:
            session: tf.Session()
            dir_model: dir with weights
        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.session, dir_model)


    def save_session(self):
        """Saves session = weights as a checkpoint"""

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(self.FLAGS.output_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        current_step = tf.train.global_step(self.session, self.global_step)
        self.saver.save(self.session, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))

    def close_session(self):
        """Closes the session"""
        self.session.close()



    def initialize_summaries(self):
        """Defines variables for Tensorboard
        Args:
            output_dir: (string) where the results are written
        """
        
        # Summaries: gradient values, loss and accuracy
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                var_name = v.name.replace(':','_')
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(var_name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(var_name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(self.FLAGS.output_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.session.graph)

        # Valid summaries
        self.valid_summary_op = tf.summary.merge([loss_summary, acc_summary])
        valid_summary_dir = os.path.join(self.FLAGS.output_dir, "summaries", "valid")
        self.valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, self.session.graph)




    def add_summary(self):
        """Defines variables for Tensorboard
        Args:
            dir_output: (string) where the results are written
        """
        merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.FLAGS.output_dir, self.session.graph)





    def train_step(self, x_train_batch, y_train_batch):
        """
        A single training step
        """
        feed_dict = {
        self.input_x: x_train_batch,
        self.input_y: y_train_batch,
        self.dropout_keep_prob: self.FLAGS.dropout_keep_prob
        }


        if self.FLAGS.summary:
            _, step, summaries, loss, accuracy = self.session.run(
                [self.optimizer, self.global_step, self.train_summary_op, self.loss, self.accuracy],feed_dict)
                    
            self.train_summary_writer.add_summary(summaries, step)
        else:
            _, step, loss = self.session.run(
                [self.optimizer, self.global_step, self.loss],feed_dict)

        if (step + 1) % 10 == 0:
            #epoch = ( step // num_batches_per_epoch) + 1
            #relative_step = (step % num_batches_per_epoch) + 1
            #time_str = datetime.datetime.now().isoformat()
            #print("{}: epoch {}/{}, step {}/{}, loss {:g}".format(time_str, epoch, FLAGS.num_epochs, relative_step, num_batches_per_epoch, loss))
            print("step---------->")





    def valid_step(self, x_valid, y_valid):
        """
        Evaluates model on a validation set
        """

        print("\nEvaluation:")
        batch_size = self.FLAGS.batch_size
        valid_batches = load_utils.batch_iter(list(zip(x_valid, y_valid)), batch_size, 1)
        num_valid_batches = (len(x_valid) - 1) // batch_size + 1

        sum_accuracy = 0
        model.confusion.load(np.zeros([num_classes,num_classes]))
        for valid_batch in valid_batches:
            x_valid_batch, y_valid_batch = zip(*valid_batch)

            feed_dict = {
                self.input_x: x_valid_batch,
                self.input_y: y_valid_batch,
                self.dropout_keep_prob: 1.0
            }

            if FLAGS.summary:
                step, summaries, loss, accuracy, cnf_matrix = self.session.run(
                    [self.global_step, self.dev_summary_op, self.loss, self.accuracy, self.confusion_update], feed_dict)

                self.writer.add_summary(summaries, step)
            else:
                step, loss, accuracy, cnf_matrix = self.session.run(
                    [self.global_step, self.loss, self.accuracy, self.confusion_update], feed_dict)    

            sum_accuracy += accuracy

        valid_accuracy = sum_accuracy / num_valid_batches

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, valid_accuracy {:g}".format(time_str, step, valid_accuracy))
        print("Confusion matrix:")
        print(cnf_matrix)

        return valid_accuracy
