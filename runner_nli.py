import tensorflow as tf
import vis_utils
from data_utils import *
import utils
import models_nli as models   # This is necessary!
import model_utils
import time


def run(args, ckpt_dir, ckpt_file):

    # Loading data
    print("Loading data")
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
        best_acc = 0
        wait = 0
        steps_per_epoch = train_x.shape[0] // args.step_size // args.batch_size - 1
        early_stop = False
        for i in range(args.epochs):
            if early_stop:
                break
            for j in range(steps_per_epoch):
                print("\n")
                print("Epoch " + str(i) + " Step " + str(j) + " at " + str(time.ctime()))
                step_loss, step_accuracy = model.train_for_step(sess, train_x, train_y, j*args.step_size*args.batch_size, args.step_size)
                print('Batch: ' + str(j*args.step_size) + '  loss: ' + str(step_loss) + '  acc: ' + str(step_accuracy))
                # print('Train accuracy = ', model.evaluate_accuracy(sess, train_x, train_y))
                # print(sess.run(tf.all_variables()[0][0]))

                dev_idx = np.random.choice(dev_x.shape[0], size=dev_x.shape[0]//10, replace=False)
                dev_x_sample = dev_x[dev_idx]
                dev_y_sample = dev_y[dev_idx]
                dev_acc = model.evaluate_accuracy(sess, dev_x_sample, dev_y_sample)
                print('SNLI Dev accuracy estimation = ', dev_acc)


                if dev_acc > best_acc:
                    wait = 0
                    best_acc = dev_acc
                    print("Checkpointing with Best SNLI Dev accuracy ", dev_acc)
                    saver.save(sess, ckpt_file)
                else:
                    wait += 1

                if j % 10 == 0:
                    print('Dev matched accuracy', model.evaluate_accuracy(sess, dev_matched_x, dev_matched_y))
                    print('Dev mismatched accuracy', model.evaluate_accuracy(sess, dev_mismatched_x, dev_mismatched_y))
                    dev_acc = model.evaluate_accuracy(sess, dev_x, dev_y)
                    print('SNLI Dev accuracy', dev_acc)

                if wait > args.patience:
                    print("No improvements in the last " + str(wait) + " steps, stopped early.")
                    early_stop = True
                    break

        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        print("Finished")

    if model.use_alphas:
        print("Producing visualization...")
        htmls = vis_utils.knit_nli(test_x, test_y, word_dict, None, model, sess, args.vis_num)
        htmls.extend(vis_utils.knit_nli(dev_matched_x, dev_matched_y, word_dict, None, model, sess, args.vis_num))
        htmls.extend(vis_utils.knit_nli(dev_mismatched_x, dev_mismatched_y, word_dict, None, model, sess, args.vis_num))

        f = open(os.path.join(ckpt_dir, "vis.html"), "wb")
        for i in htmls:
            f.write(i)
        f.close()
        print("...done")

