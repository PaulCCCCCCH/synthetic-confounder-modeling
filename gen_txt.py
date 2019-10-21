import pickle

f = open("out/samples.pkl", "rb")
samples = pickle.load(f)
f.close()

f = open("out/samples.txt", "wb")
for i in samples:
    line = str(i["label"]) + " " + " ".join(i["sentence"]) + "\n"
    f.write(line)
f.close()
