import gensim
import logging
import copy
import numpy as np
from visdom import Visdom


class Coherence(object):
    def __init__(self, corpus=None, texts=None, dictionary=None, coherence=None, window_size=None, topn=None, logger=None, viz_env=None, title=None):
        self.corpus = corpus
        self.dictionary = dictionary
        self.metric = coherence
        self.texts = texts
        self.window_size = window_size
        self.topn = topn
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        model = None
        topics = None
        if 'model' in kwargs:
            model = kwargs['model']
        if 'topics' in kwargs:
            topics = kwargs['topics']
        cm = gensim.models.CoherenceModel(model, topics, self.texts, self.corpus, self.dictionary, self.window_size, self.metric, self.topn)
        return cm.get_coherence()


class Perplexity(object):
    def __init__(self, corpus=None, logger=None, viz_env=None, title=None):
        self.corpus = corpus
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        model = kwargs['model']
        corpus_words = sum(cnt for document in self.corpus for _, cnt in document)
        perwordbound = model.bound(self.corpus) / corpus_words
        return np.exp2(-perwordbound)

        
class Diff(object):
    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, normed=True, logger=None, viz_env=None, title=None):
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        model = kwargs['model']
        other_model = kwargs['other_model']
        diff_matrix, _ = model.diff(other_model, self.distance, self.num_words, self.n_ann_terms, self.normed)
        return np.diagonal(diff_matrix)


class Convergence(object):
    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, normed=True, logger=None, viz_env=None, title=None):
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        model = kwargs['model']
        other_model = kwargs['other_model']
        diff_matrix, _ = model.diff(other_model, self.distance, self.num_words, self.n_ann_terms, self.normed)
        return np.sum(np.diagonal(diff_matrix))
        

class Callback(object):
    def __init__(self, metrics=None):
        # list of metrics to be plot
        self.metrics = metrics

    def set_model(self, model):
        self.model = model
        self.previous = None
        # check for any metric that need model state from previous epoch
        if any(isinstance(metric, (Diff, Convergence)) for metric in self.metrics):
            self.previous = copy.deepcopy(model)
        # set Visdom instance for visualization in current topic model
        self.viz = Visdom()
        # store initial plot windows of every metric (same window will be updated with increasing epochs) 
        self.windows = []
        # set logger for current topic model
        model_type = type(self.model).__name__
        self.log_type = logging.getLogger(model_type)

    def on_epoch_end(self, epoch, topics=None):
        # provide topics after current epoch if coherence is not valid for this topic model
        for i, metric in enumerate(self.metrics):
            value = metric.get_value(topics=topics, model=self.model, other_model=self.previous)
            if value.ndim>0:
                self.previous = copy.deepcopy(self.model)

            if metric.logger=="visdom":
                if epoch==0:
                    if value.ndim>0:
                        self.diff_mat = np.array([value])
                        viz_metric = self.viz.heatmap(X=self.diff_mat.T, env=metric.viz_env, opts=dict(xlabel='Epochs', ylabel=type(metric).__name__))
                        self.windows.append(copy.deepcopy(viz_metric)) 
                    else:
                        viz_metric = self.viz.line(Y=np.array([value]), X=np.array([epoch]), env=metric.viz_env, opts=dict(xlabel='Epochs', ylabel=type(metric).__name__))
                        self.windows.append(copy.deepcopy(viz_metric))                        
                else:
                    if value.ndim>0:
                        self.diff_mat = np.concatenate((self.diff_mat, np.array([value])))
                        self.viz.heatmap(X=self.diff_mat.T, env=metric.viz_env, win=self.windows[i], opts=dict(xlabel='Epochs', ylabel=type(metric).__name__))
                    else:
                        self.viz.updateTrace(Y=np.array([value]), X=np.array([epoch]), env=metric.viz_env, win=self.windows[i])
                        
            if metric.logger=='shell':
                statement = ' '.join((type(metric).__name__, "estimate:", str(value)))
                self.log_type.info(statement)


