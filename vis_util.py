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
MAP = {
    0: "negative",
    1: "positive"
    }

def html_render(x_orig, alphas, max_len=30):
    epsilon = 1e-10
    k = 80
    b = 600
    #color_vals = (100 + np.log(alphas)) * 40
    color_vals = k * np.log(alphas + epsilon) + b
    #x_orig_words = x_orig.split(' ')[:max_len]
    

    x_orig_words = x_orig[:max_len]

    orig_html = []
    for i in range(len(x_orig_words)):
        color_val = color_vals[i]
        colors = [0, 0, 0]
        if color_val >= 510:
            colors = [255, 0, 0]
        elif color_val >= 255:
            colors = [int(color_val - 255), 0, 0]
        else:
            colors = [0, 255 - int(color_val), 0]

        orig_html.append(format("<b style='color:rgb(%d,%d,%d)'>%s</b>" %(colors[0], colors[1], colors[2], x_orig_words[i])))
    
    orig_html = ' '.join(orig_html)
    return orig_html


def knit(xs, ys, word_dict, model, sess, show=100):

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

        line += html_render([inv_dict[w] for w in x], alphas)
        line += "</p>"
        line += str(alphas)
        line += "<p>"
        line += "###############################"
        line += "</p>"
        html_content.append(line)
    return html_content

