import logging
import gensim
from gensim.models import ldamodel
from gensim.corpora.dictionary import Dictionary

from gensim.models.callbacks import CoherenceCallback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

texts = [['bank','river','shore','water'],
        ['river','water','flow','fast','tree'],
        ['bank','water','fall','flow'],
        ['bank','bank','water','rain','river'],
        ['river','water','mud','tree'],
        ['money','transaction','bank','finance'],
        ['bank','borrow','money'], 
        ['bank','finance'],
        ['finance','money','sell','bank'],
        ['borrow','sell'],
        ['bank','loan','sell']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

callbacks = [CoherenceCallback(corpus=corpus, coherence="u_mass", texts=texts, window_size=10, logger='visdom', viz_env='LdaModel')]

# training LDA model
model = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, passes=10, callbacks=callbacks)

# to get coherence value from anywhere else (ex. sklearn score)
print(CoherenceCallback(model=model, corpus=corpus, coherence="u_mass").get_metric())


