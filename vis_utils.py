import pickle
import numpy as np
"""
#Loading date:
print("Loading vocabulary")
f = open("data/vocab.pkl", "rb")
word_dict = pickle.load(f)
f.close()

print("Loading dataset")
f = open("out/samples.pkl", "rb")
samples = pickle.load(f)
f.close()
"""


def html_render(x_orig, alphas, inv_dict, effect_list, max_len=30):
    epsilon = 1e-10
    k = 80
    b = 600
    #color_vals = (100 + np.log(alphas)) * 40
    color_vals = k * np.log(alphas + epsilon) + b
    #x_orig_words = x_orig.split(' ')[:max_len]
    

    x_orig = x_orig[:max_len]

    orig_html = []
    for i in range(len(x_orig)):
        color_val = color_vals[i]
        colors = [0, 0, 0]
        if color_val >= 510:
            colors = [255, 0, 0]
        elif color_val >= 255:
            colors = [int(color_val - 255), 0, 0]
        else:
            colors = [0, 255 - int(color_val), 0]

        #Show word, hover to see word effect
        #orig_html.append(format("<b style='color:rgb(%d,%d,%d)' title='%s'>%s</b>" %(colors[0], colors[1], colors[2], inv_dict[x_orig[i]], str(effect_list[x_orig[i]]))))


        #Show word effect, hover to see word
        if effect_list:
            orig_html.append(format("<b style='color:rgb(%d,%d,%d)' title='%s'>%s</b>" %(colors[0], colors[1], colors[2], inv_dict[x_orig[i]], str(effect_list[x_orig[i]]))))
        else:
            orig_html.append(format("<b style='color:rgb(%d,%d,%d)'>%s</b>" %(colors[0], colors[1], colors[2], inv_dict[x_orig[i]])))

    orig_html = ' '.join(orig_html)
    return orig_html


def knit(xs, ys, word_dict, effect_list, model, sess, show=100):
    MAP = {
        0: "negative",
        1: "positive"
    }
    inv_dict = {}
    for k in word_dict.keys():
        inv_dict[word_dict[k]] = k

    predictions = []
    alphass = []
    for i in range(show // model.batch_size - 1):
        p_batch, alpha_batch = model.predict(sess, xs[i*model.batch_size: (i+1)*model.batch_size])
        p_batch = np.argmax(p_batch, axis=1)
        predictions.extend(p_batch)
        alphass.extend(alpha_batch)

    html_content = []
    for i in range(show - model.batch_size):

        x, y = xs[i], ys[i]
        index = str(i)

        prediction, alphas = predictions[i], alphass[i]


        line = "<p>"
        line += "Sample %d ###################" %i
        line += "<p>"
        line += "<p>"
        line += MAP[prediction]
        line += " <-prediction...|...target-> "
        line += MAP[y]
        line += "<p>"

        line += html_render(x, alphas, inv_dict, effect_list)
        line += "</p>"
        line += "attention weights:"
        line += str(alphas)
        if effect_list:
            line += "<p>effect list:"
            line += str([effect_list[w] for w in x])
            line += "</p>"
        line += "<p>"
        line += "###############################"
        line += "</p>"
        html_content.append(line)
    return html_content


def knit_nli(xs, ys, word_dict, effect_list, model, sess, show=100):
    MAP = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
    inv_dict = {}
    for k in word_dict.keys():
        inv_dict[word_dict[k]] = k

    predictions = []
    alphass_hypo = []
    alphass_prem = []
    for i in range(show // model.batch_size - 1):
        p_batch, alpha_hypo_batch, alpha_prem_batch = model.predict(sess, xs[i*model.batch_size: (i+1)*model.batch_size])
        p_batch = np.argmax(p_batch, axis=1)
        predictions.extend(p_batch)
        alphass_hypo.extend(alpha_hypo_batch)
        alphass_prem.extend(alpha_prem_batch)

    html_content = []
    for i in range(show - model.batch_size):

        x_hypo, x_prem, y = xs[i, 0, :], xs[i, 1, :], ys[i]
        index = str(i)

        prediction, alphas_hypo, alphas_prem = predictions[i], alphass_hypo[i], alphass_prem[i]

        line = "<p>"
        line += "Sample %d ###################" %i
        line += "<p>"
        line += "<p>"
        line += MAP[prediction]
        line += " <-prediction...|...target-> "
        line += MAP[y]

        line += "<p>"
        line += html_render(x_hypo, alphas_hypo, inv_dict, effect_list)
        line += "</p>"
        line += "attention weights:"
        line += str(alphas_hypo)

        line += "<p>"
        line += html_render(x_prem, alphas_prem, inv_dict, effect_list)
        line += "</p>"
        line += "attention weights:"
        line += str(alphas_prem)

        line += "<p>"
        line += "###############################"
        line += "</p>"
        html_content.append(line)
    return html_content

