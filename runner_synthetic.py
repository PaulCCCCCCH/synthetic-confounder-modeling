import tensorflow as tf
import os
import vis_utils
from data_utils import load_all_data, load_file
import utils
import model_utils
import models   # This is necessary!


def run(args, ckpt_dir, ckpt_file):
    assert args.task == "synthetic"

    #Loading data:
    train_x, train_y, test_x, test_y, word_dict, effect_list, embedding_matrix = load_all_data(args)
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
        prev_init = models.AdditiveModel
        model = model_utils.get_additive_model(prev_init, pred_model, key_word_model)
    else:
        model = pred_model

    utils.initialize_uninitialized_global_variables(sess)

    print("Buidling the model. Model name: {}".format(args.modelname))


    if args.test:
        saver.restore(sess, ckpt_file)
        print('Test accuracy = ', model.evaluate_accuracy(sess, test_x, test_y))
        print('Signal capturing score= ', model.evaluate_capturing(sess, test_x, test_y, effect_list))

    else:
        sess.run(tf.assign(pred_model.embedding_w, embedding_matrix))

        if os.path.exists(ckpt_file+".meta"):
            print('Restoring Model')
            saver.restore(sess, ckpt_file)

        print('Training..')
        for i in range(args.epochs):
            epoch_loss, epoch_accuracy = model.train_for_epoch(sess, train_x, train_y)
            print(i, 'loss: ', epoch_loss, 'acc: ', epoch_accuracy)
            #print('Train accuracy = ', model.evaluate_accuracy(sess, train_x, train_y))
            #print(sess.run(tf.all_variables()[0][0]))
            print('Test accuracy = ', model.evaluate_accuracy(sess, test_x, test_y))
            if model.use_alphas:
                print('Signal capturing score= ', model.evaluate_capturing(sess, test_x, test_y, effect_list))
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        print("Saving the model")
        saver.save(sess, ckpt_file)
        print("Finished")

    if model.use_alphas:
        print("Producing visualization")
        htmls = vis_utils.knit(test_x, test_y, word_dict, effect_list, model, sess, 100)
        f = open(os.path.join(ckpt_dir, "vis.html"), "wb")
        for i in htmls:
            f.write(i)
        f.close()



