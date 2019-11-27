import tensorflow as tf
import vis_utils
from data_utils import *
import utils
import models_nli as models   # This is necessary!
import model_utils


def run(args, ckpt_dir, ckpt_file):

    # Loading data
    if args.task == "snli":
        train_x, train_y, dev_x, dev_y, test_x, test_y, word_dict, embedding_matrix = load_all_data_snli(args)
        dev_matched_x, dev_matched_y, dev_mismatched_x, dev_mismatched_y = load_test_data_mnli(args, word_dict)
    elif args.task == "mnli":
        train_x, train_y, dev_matched_x, dev_matched_y, dev_mismatched_x, dev_mismatched_y, word_dict, embedding_matrix = load_all_data_mnli(args)
        dev_x, dev_y, test_x, test_y = load_test_data_snli(args, word_dict)
    else:
        raise NotImplementedError

    vocab_size = embedding_matrix.shape[0]

    print("Dataset building all done")

    sess = tf.Session()
    use_additive = False
    if args.kwm_path != "":

        prev_arg_file = os.path.join(args.kwm_path, "args.pkl")
        prev_args = load_file(prev_arg_file)

        print("Loading key-word model with the following parameters: ")
        print(prev_args.__dict__)

        with tf.variable_scope(prev_args.modelname) as scope:
            prev_init = eval(model_utils.all_models[prev_args.modeltype])
            key_word_model = model_utils.get_model(prev_args, prev_init, vocab_size)
        kwm_saver = tf.train.Saver()

        kwm_ckpt = os.path.join(args.kwm_path, prev_args.modelname)
        kwm_saver.restore(sess, kwm_ckpt)
        use_additive = True

    with tf.variable_scope(args.modelname) as scope:
        init = eval(model_utils.all_models[args.modeltype])
        pred_model = model_utils.get_model(args, init, vocab_size)

    saver = tf.train.Saver()

    if use_additive:
        init = models.AdditiveModel
        model = model_utils.get_additive_model(init, pred_model, key_word_model)
    else:
        model = pred_model

    utils.initialize_uninitialized_global_variables(sess)

    print("Building the model. Model name: {}".format(args.modelname))

    if args.test:
        saver.restore(sess, ckpt_file)
        print('SNLI test accuracy = ', model.evaluate_accuracy(sess, test_x, test_y))
        print('Matched dev accuracy = ', model.evaluate_accuracy(sess, dev_matched_x, dev_matched_y))
        print('Mismatched dev accuracy = ', model.evaluate_accuracy(sess, dev_mismatched_x, dev_mismatched_y))

    else:
        sess.run(tf.assign(pred_model.embedding_w, embedding_matrix))

        if os.path.exists(ckpt_file+".meta"):
            print('Restoring Model')
            saver.restore(sess, ckpt_file)

        print('Training..')
        for i in range(args.epochs):
            epoch_loss, epoch_accuracy = model.train_for_epoch(sess, train_x, train_y)
            print(i, 'loss: ', epoch_loss, 'acc: ', epoch_accuracy)
            # print('Train accuracy = ', model.evaluate_accuracy(sess, train_x, train_y))
            # print(sess.run(tf.all_variables()[0][0]))
            print('SNLI Dev accuracy = ', model.evaluate_accuracy(sess, dev_x, dev_y))
            print('Dev matched accuracy = ', model.evaluate_accuracy(sess, dev_matched_x, dev_matched_y))
            print('Dev mismatched accuracy = ', model.evaluate_accuracy(sess, dev_mismatched_x, dev_mismatched_y))
            saver.save(sess, ckpt_file)

        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        print("Saving the model")
        saver.save(sess, ckpt_file)
        print("Finished")

    if model.use_alphas:
        print("Producing visualization")
        htmls = vis_utils.knit_nli(test_x, test_y, word_dict, None, model, sess, 100)
        htmls.extend(vis_utils.knit_nli(dev_matched_x, dev_matched_y, word_dict, None, model, sess, 100))
        htmls.extend(vis_utils.knit_nli(dev_mismatched_x, dev_mismatched_y, word_dict, None, model, sess, 100))

        f = open(os.path.join(ckpt_dir, "vis.html"), "wb")
        for i in htmls:
            f.write(i)
        f.close()


