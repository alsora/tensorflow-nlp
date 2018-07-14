import os
import shutil
import datetime
import numpy as np
import tensorflow as tf
import data_helpers.load as load_utils
from logger_utils import get_logger
import tf_helpers.hyperparams_flags

class BaseModel(object):
    """Generic base class for neural network models"""

    def __init__(self, FLAGS):

        self.FLAGS = FLAGS
        self.logger = get_logger(os.path.join(self.FLAGS.model_dir, "log.txt"))
        self.session = None
        self.saver = None
        
        self.overwrite_hyperparams()


    def overwrite_hyperparams(self):
        """Overwrite default hyperparameters of a network, based on the flags"""
        try:
            default_hyperparams = self.hyperparams
            for key in default_hyperparams:
                try:
                    flag = self.FLAGS[key]
                    param_value = flag.value
                    if param_value is not None:
                        self.hyperparams[key] = param_value
                except:
                    pass
        except:
            pass


    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        session_conf = tf.ConfigProto(
            allow_soft_placement=self.FLAGS.allow_soft_placement,
            log_device_placement=self.FLAGS.log_device_placement)
        self.session = tf.Session(config=session_conf)
        self.session.run(tf.global_variables_initializer())
        try:  
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS.num_checkpoints)
        except:
            pass


    def restore_session(self, dir_model):
        """Reload weights into session
        Args:
            session: tf.Session()
            dir_model: dir with weights
        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.session, dir_model)

    
    def save_model(self, output_folder = ''):

        if not output_folder:
            output_folder = os.path.join(self.FLAGS.model_dir, "saved")

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        builder = tf.saved_model.builder.SavedModelBuilder(output_folder)
        builder.add_meta_graph_and_variables(
            self.session,
            [tf.saved_model.tag_constants.SERVING],
            clear_devices=True)
        
        builder.save()


    def restore_saved_model(self, model_dir, tag = [tf.saved_model.tag_constants.SERVING]):

        saved_model_dir = os.path.join(model_dir, "saved")

        tf.saved_model.loader.load(self.session, tag, saved_model_dir)

        
    def save_session(self):
        """Saves session = weights as a checkpoint"""

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(self.FLAGS.model_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        current_step = tf.train.global_step(self.session, self.global_step)
        path = self.saver.save(self.session, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))


    def close_session(self):
        """Closes the session"""
        self.session.close()


    def initialize_summaries(self):
        """Defines variables for Tensorboard
        Args:
            model_dir: (string) where the results are written
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
        train_summary_dir = os.path.join(self.FLAGS.model_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.session.graph)

        # Valid summaries
        self.valid_summary_op = tf.summary.merge([loss_summary, acc_summary])
        valid_summary_dir = os.path.join(self.FLAGS.model_dir, "summaries", "valid")
        self.valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, self.session.graph)


    def add_summary(self):
        """Defines variables for Tensorboard
        Args:
            dir_output: (string) where the results are written
        """
        merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.FLAGS.model_dir, self.session.graph)


    def init_dataset(self, feed_dict):
        dataset_op = self.session.graph.get_operation_by_name("dataset_init")
        self.session.run(dataset_op, feed_dict=feed_dict)


    def train_step(self, x_train, y_train):
        """
        Trains model on a train set
        """

        input_x_op = self.session.graph.get_operation_by_name("input_x").outputs[0]
        input_y_op = self.session.graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob_op = self.session.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        global_step_op = self.session.graph.get_operation_by_name("global_step").outputs[0]

        optimizer_op = self.session.graph.get_operation_by_name("loss/optimizer").outputs[0]
        loss_op = self.session.graph.get_operation_by_name("loss/loss").outputs[0]

        d_ = {
        input_x_op: x_train,
        input_y_op: y_train
        }

        self.init_dataset(d_)

        train_batches_per_epoch = (len(x_train) - 1) // self.FLAGS.batch_size + 1

        sum_loss = 0
        for current_step in range (train_batches_per_epoch):

            if self.FLAGS.summary:
                _, step, summaries, loss = self.session.run(
                    [optimizer_op, global_step_op, self.train_summary_op, loss_op], feed_dict={dropout_keep_prob_op: self.hyperparams['dropout_keep_prob']})
                        
                self.train_summary_writer.add_summary(summaries, step)
            else:
                _, step, loss = self.session.run(
                    [optimizer_op, global_step_op, loss_op], feed_dict={dropout_keep_prob_op: self.hyperparams['dropout_keep_prob']})
            
            sum_loss += loss

            time_str = datetime.datetime.now().isoformat()
            if (current_step + 1) % 10 == 0:
                print("{}: step {}/{}, loss {:g}".format(time_str, current_step + 1, train_batches_per_epoch, loss))

        mean_loss = sum_loss/ train_batches_per_epoch

        return mean_loss


    def test_step(self, x_test, y_test):
        """
        Evaluates model on a validation set
        """

        print("Evaluation:")

        input_x_op = self.session.graph.get_operation_by_name("input_x").outputs[0]
        input_y_op = self.session.graph.get_operation_by_name("input_y").outputs[0]
        global_step_op = self.session.graph.get_operation_by_name("global_step").outputs[0]

        loss_op = self.session.graph.get_operation_by_name("loss/loss").outputs[0]

        predictions_op = self.session.graph.get_operation_by_name("output/predictions").outputs[0] 

        accuracy_op = self.session.graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        confusion_update_op = self.session.graph.get_operation_by_name("accuracy/confusion_update").outputs[0]

        d_ = {
        input_x_op: x_test,
        input_y_op: y_test
        }

        self.init_dataset(d_)

        valid_batches_per_epoch = (len(x_test) - 1) // self.FLAGS.batch_size + 1

        sum_accuracy = 0
        
        confusion_variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="accuracy/confusion")[0]
        self.session.run([confusion_variable.initializer])

        for current_step in range(valid_batches_per_epoch):

            if self.FLAGS.summary:
                step, summaries, loss, accuracy, cnf_matrix, predictions = self.session.run(
                    [global_step_op, self.dev_summary_op, loss_op, accuracy_op, confusion_update_op, predictions_op])

                self.writer.add_summary(summaries, step)
            else:
                step, loss, accuracy, cnf_matrix, predictions = self.session.run(
                    [global_step_op, loss_op, accuracy_op, confusion_update_op, predictions_op])    

            sum_accuracy += accuracy

            try:
                all_predictions = np.concatenate((all_predictions, predictions), axis=0)
            except NameError:
                all_predictions = predictions


        valid_accuracy = sum_accuracy / valid_batches_per_epoch

        time_str = datetime.datetime.now().isoformat()
        print("{}: valid_accuracy {:g}".format(time_str, valid_accuracy))
        print("Confusion matrix:")
        print(cnf_matrix)

        return valid_accuracy, all_predictions


    def predict_step(self, x):
        """
        Predict labels for data x 
        """

        input_x = self.session.graph.get_operation_by_name("input_x").outputs[0]
        predictions_op = self.session.graph.get_operation_by_name("output/predictions").outputs[0] 

        d_ = {
        input_x: x
        }

        self.init_dataset(d_)

        return self.session.run([predictions_op])

