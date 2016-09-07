import nltk
import pickle
from string import punctuation
from random import shuffle

def convert_to_universal_tag(t, reverse=False):
    tagdict = {
        'n': "NOUN",
        'num': "NUM",
        'v-fin': "VERB",
        'v-inf': "VERB",
        'v-ger': "VERB",
        'v-pcp': "VERB",
        'pron-det': "PRON",
        'pron-indp': "PRON",
        'pron-pers': "PRON",
        'art': "DET",
        'adv': "ADV",
        'conj-s': "CONJ",
        'conj-c': "CONJ",
        'conj-p': "CONJ",
        'adj': "ADJ",
        'ec': "PRT",
        'pp': "ADP",
        'prp': "ADP",
        'prop': "NOUN",
        'pro-ks-rel': "PRON",
        'proadj': "PRON",
        'prep': "ADP",
        'nprop': "NOUN",
        'vaux': "VERB",
        'propess': "PRON",
        'v': "VERB",
        'vp': "VERB",
        'in': "X",
        'prp-': "ADP",
        'adv-ks': "ADV",
        'dad': "NUM",
        'prosub': "PRON",
        'tel': "NUM",
        'ap': "NUM",
        'est': "NOUN",
        'cur': "X",
        'pcp': "VERB",
        'pro-ks': "PRON",
        'hor': "NUM",
        'pden': "ADV",
        'dat': "NUM",
        'kc': "ADP",
        'ks': "ADP",
        'adv-ks-rel': "ADV",
        'npro': "NOUN",
    }
    if t in ["N|AP","N|DAD","N|DAT","N|HOR","N|TEL"]:
        t = "NUM"
    if reverse:
        if "|" in t: t = t.split("|")[0]
    else:
        if "+" in t: t = t.split("+")[1]
        if "|" in t: t = t.split("|")[1]
        if "#" in t: t = t.split("#")[0]
    t = t.lower()
    return tagdict.get(t, "." if all(tt in punctuation for tt in t) else t)

# nltk.corpus.mac_morpho.tagged_sents is incorrect, converting tagged_paras to tagged_sents
dataset1 = list(nltk.corpus.floresta.tagged_sents())
dataset2 = [[w[0] for w in sent] for sent in nltk.corpus.mac_morpho.tagged_paras()]

traindata = [[(w, convert_to_universal_tag(t)) for (w, t) in sent] for sent in dataset1]
traindata2 = traindata + [[(w, convert_to_universal_tag(t, reverse=True)) for (w, t) in sent] for sent in dataset2]

shuffle(traindata)
shuffle(traindata2)

regex_patterns = [
    (r"^[nN][ao]s?$", "ADP"),
    (r"^[dD][ao]s?$", "ADP"),
    (r"^[pP]el[ao]s?$", "ADP"),
    (r"^[nN]est[ae]s?$", "ADP"),
    (r"^[nN]um$", "ADP"),
    (r"^[nN]ess[ae]s?$", "ADP"),
    (r"^[nN]aquel[ae]s?$", "ADP"),
    (r"^\xe0$", "ADP"),
]

tagger = nltk.BigramTagger(
            traindata, backoff=nltk.RegexpTagger(
                regex_patterns, backoff=nltk.UnigramTagger(
                    traindata2, backoff=nltk.AffixTagger(
                        traindata2, backoff=nltk.DefaultTagger('NOUN')
                    )
                )
            )
        )
templates = nltk.brill.fntbl37()
tagger = nltk.BrillTaggerTrainer(tagger, templates)
tagger = tagger.train(traindata, max_rules=100)

with open("tagger.pkl", "wb") as f:
    pickle.dump(tagger, f)

