import logging
import gensim
from gensim.models import ldamodel
from gensim.corpora.dictionary import Dictionary

from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric

import pandas as pd
import re
import smart_open
import random

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# texts = [['bank','river','shore','water'],
#         ['river','water','flow','fast','tree'],
#         ['bank','water','fall','flow'],
#         ['bank','bank','water','rain','river'],
#         ['river','water','mud','tree'],
#         ['money','transaction','bank','finance'],
#         ['bank','borrow','money'], 
#         ['bank','finance'],
#         ['finance','money','sell','bank'],
#         ['borrow','sell'],
#         ['bank','loan','sell']]

dataframe = pd.read_csv('movie_plots.csv')
texts = []
for line in dataframe.Plots:
    lowered = line.lower()
    words = re.findall(r'\w+', lowered, flags = re.UNICODE | re.LOCALE)
    texts.append(words)

training_texts = texts[:500]
validation_texts = texts[500:750]
test_texts = texts[750:]

dictionary = Dictionary(training_texts)

training_corpus = [dictionary.doc2bow(text) for text in training_texts]
validation_corpus = [dictionary.doc2bow(text) for text in validation_texts]
test_corpus = [dictionary.doc2bow(text) for text in test_texts]

pl_validation = PerplexityMetric(corpus=validation_corpus, logger="visdom", viz_env="LdaModel", title="Perplexity")
pl_test = PerplexityMetric(corpus=test_corpus, logger="visdom", viz_env="LdaModel", title="Perplexity")

ch_umass_training = CoherenceMetric(corpus=training_corpus, coherence="u_mass", texts=training_texts, logger="visdom", viz_env="LdaModel", title="Coherence")
ch_cv_training = CoherenceMetric(corpus=training_corpus, coherence="c_v", texts=training_texts, logger="visdom", viz_env="LdaModel", title="Coherence")

diff_kl = DiffMetric(distance="kulback_leibler", logger="visdom", viz_env="LdaModel", title="Diff")
diff_jc = DiffMetric(distance="jaccard", logger="visdom", viz_env="LdaModel", title="Diff")

convergence_kl = ConvergenceMetric(distance="kulback_leibler", logger="visdom", viz_env="LdaModel", title="Convergence")
convergence_jc = ConvergenceMetric(distance="jaccard", logger="visdom", viz_env="LdaModel", title="Convergence")

callbacks = [pl_validation, pl_test, ch_umass_training, ch_cv_training, diff_kl, diff_jc, convergence_kl, convergence_jc]


# training LDA model
model = ldamodel.LdaModel(corpus=training_corpus, id2word=dictionary, passes=5, num_topics=5, callbacks=callbacks)


# to get coherence value from anywhere else (ex. sklearn score)
# print(Coherence(coherence="u_mass", texts=texts).get_value(model=model))
