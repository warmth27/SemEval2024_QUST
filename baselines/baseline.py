import numpy as np
import torch
import json, os
from sklearn.svm import SVC
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from model.models import *
from skorch import NeuralNetClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils.class_weight import compute_class_weight
from skorch.callbacks import Checkpoint, EarlyStopping
from skorch.dataset import ValidSplit

# lan = ["en", "fr", "ge", "it", "po", "ru"]
lan = ["en"]

USE_CUDA = torch.cuda.is_available()  # Set this to False if you don't want to use CUDA

model_name = ["bert-base-cased", "roberta-base", "xlnet-base-cased"]

m = model_name[0]

for l in lan:
    print("Processing {}".format(l))
    with open('../train/{}/train-articles-subtask-1-{}-{}.json'.format(l, m, l), 'r') as inf:
        l_data = json.load(inf)
        x_nn = []
        x_clf = []
        x_text = []
        y = []
        for idx, line in enumerate(l_data):
            embed = line[m]
            text = line['text']
            label = line['label']
            x_clf.append(np.mean(embed, axis=0))
            x_nn.append(embed)
            x_text.append(text)
            y.append(label)
            print("Processed {} of {} lines".format(idx+1, len(l_data)))

        weights = compute_class_weight('balanced', classes=np.unique(y),  y=np.array(y)).tolist()

        ckpoint = Checkpoint(monitor='valid_acc_best', load_best=True)
        es = EarlyStopping(monitor='valid_acc_best', patience=10)
        batch_size = 128
        epoch = 50
        vsplit = 60
        net = NeuralNetClassifier(
            Simple_CNN,
            max_epochs=epoch,
            lr=0.001,
            device=('cuda' if USE_CUDA else 'cpu'),
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights)),
            train_split=ValidSplit(vsplit, True),
            callbacks=[ckpoint, es]
        )

        net2 = NeuralNetClassifier(
            LSTM_ATT(batch_size=batch_size, output_size=3, hidden_size=50, embedding_length=768),
            max_epochs=epoch,
            lr=0.001,
            device=('cuda' if USE_CUDA else 'cpu'),
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights)),
            train_split=ValidSplit(vsplit, True),
            batch_size=batch_size,
            callbacks=[ckpoint, es]
        )

        net3 = NeuralNetClassifier(
            SelfAttention(batch_size=batch_size, output_size=3, hidden_size=50, embedding_length=768),
            max_epochs=epoch,
            lr=0.001,
            device=('cuda' if USE_CUDA else 'cpu'),
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights)),
            train_split=ValidSplit(vsplit, True),
            batch_size=batch_size,
            callbacks=[ckpoint, es]
        )

        net4 = NeuralNetClassifier(
            RCNN(batch_size=batch_size, output_size=3, hidden_size=50, embedding_length=768),
            max_epochs=epoch,
            lr=0.001,
            device=('cuda' if USE_CUDA else 'cpu'),
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights)),
            train_split=ValidSplit(vsplit, True),
            batch_size=batch_size,
            callbacks=[ckpoint, es]
        )

        random_state = 1


        clf_cnn = make_pipeline_imb(net, verbose=1)
        clf_rnn = make_pipeline_imb(net2, verbose=1)
        clf_self = make_pipeline_imb(net3, verbose=1)
        clf_rcnn = make_pipeline_imb(net4, verbose=1)

        clf_text = make_pipeline_imb(CountVectorizer(), RandomUnderSampler(), SVC(probability=True), verbose=1)

        clf_svc = make_pipeline_imb(RandomUnderSampler(), SVC(probability=True), verbose=1)
        clf_lg = make_pipeline_imb(RandomUnderSampler(), LogisticRegression(max_iter=500), verbose=1)
        clf_rf = make_pipeline_imb(RandomUnderSampler(), RandomForestClassifier(n_estimators=500), verbose=1)
        clf_nb = make_pipeline_imb(RandomUnderSampler(), GaussianNB(), verbose=1)

        x_nn = np.array(x_nn).astype(np.float32)
        y= np.array(y).astype(np.int64)

        clf_rnn.fit(x_nn, y)
        clf_self.fit(x_nn, y)
        clf_rcnn.fit(x_nn, y)
        clf_cnn.fit(x_nn, y)

        clf_text.fit(x_text, y)

        clf_svc.fit(x_clf, y)
        clf_lg.fit(x_clf, y)
        clf_rf.fit(x_clf, y)
        clf_nb.fit(x_clf,y)

        with open("../dev/subtask-1/dev-articles-subtask-1-{}-en.json".format(m), 'r') as inf2:
            dev_data = json.load(inf2)
            dev_x_clf = []
            dev_x_text = []
            dev_x_nn = []
            dev_ids = []
            for line in dev_data:
                embed = line[m]
                text = line['text']
                id = line['id']
                dev_x_clf.append(np.mean(embed, axis=0))
                dev_x_nn.append(embed)
                dev_x_text.append(text)
                dev_ids.append(id)
            dev_x_nn = np.array(dev_x_nn).astype(np.float32)


            pred_cnn = clf_cnn.predict(dev_x_nn)
            pred_rnn = clf_rnn.predict(dev_x_nn)
            pred_rcnn = clf_rcnn.predict(dev_x_nn)
            pred_self = clf_self.predict(dev_x_nn)

            pred_svc = clf_svc.predict(dev_x_clf)
            pred_lg = clf_lg.predict(dev_x_clf)
            pred_rf = clf_rf.predict(dev_x_clf)
            pred_nb = clf_nb.predict(dev_x_clf)

            pred_text = clf_text.predict(dev_x_text)

            pred_probabilities = (clf_nb.predict_proba(dev_x_clf) + clf_text.predict_proba(dev_x_text) +
                                  clf_cnn.predict_proba(dev_x_nn) + clf_svc.predict_proba(dev_x_clf) +
                                  clf_lg.predict_proba(dev_x_clf) + clf_rnn.predict_proba(dev_x_nn) +
                                  clf_self.predict_proba(dev_x_nn) + clf_rcnn.predict_proba(dev_x_nn) +
                                  clf_rf.predict_proba(dev_x_clf)) / 9

            pred_all = np.argmax(pred_probabilities, axis=-1)

            if not os.path.exists("../results/{}/{}/".format(l, m)):
                os.mkdir("../results/{}/{}/".format(l, m))

            path = "../results/{}/{}/".format(l, m)

            with open(path + m + "-rf-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_rf[i] == 0:
                        outf.write("reporting")
                    elif pred_rf[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")

            with open(path + m + "-rcnn-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_rcnn[i] == 0:
                        outf.write("reporting")
                    elif pred_rcnn[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")

            with open(path + m + "-self-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_self[i] == 0:
                        outf.write("reporting")
                    elif pred_self[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")

            with open(path + m + "-nb-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_nb[i] == 0:
                        outf.write("reporting")
                    elif pred_nb[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")

            with open(path + m + "-counter-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_text[i] == 0:
                        outf.write("reporting")
                    elif pred_text[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")

            with open(path + m + "-cnn-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_cnn[i] == 0:
                        outf.write("reporting")
                    elif pred_cnn[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")

            with open(path + m + "-rnn-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_rnn[i] == 0:
                        outf.write("reporting")
                    elif pred_rnn[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")

            with open(path + m + "-svc-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_svc[i] == 0:
                        outf.write("reporting")
                    elif pred_svc[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")

            with open(path + m + "-lg-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_lg[i] == 0:
                        outf.write("reporting")
                    elif pred_lg[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")

            with open(path + m + "-avg-en.txt", 'w') as outf:
                for i in range(len(dev_ids)):
                    outf.write(str(dev_ids[i]))
                    outf.write("\t")
                    if pred_all[i] == 0:
                        outf.write("reporting")
                    elif pred_all[i] == 1:
                        outf.write("opinion")
                    else:
                        outf.write("satire")
                    outf.write("\n")