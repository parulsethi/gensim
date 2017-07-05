import gensim
import logging
import numpy as np
from visdom import Visdom


class CoherenceCallback(object):
    def __init__(self, model=None, topics=None, texts=None, corpus=None, dictionary=None, coherence=None, window_size=None, topn=None, logger=None, viz_env=None):
        self.model = model
        self.topics = topics
        self.corpus = corpus
        self.dictionary = dictionary
        self.metric = coherence
        self.texts = texts
        self.window_size = window_size
        self.topn = topn
        self.logger = logger
        self.viz_env = viz_env

    def get_metric(self, topics=None):
        if topics is None:
            topics = self.topics
        cm = gensim.models.CoherenceModel(self.model, topics, self.texts, self.corpus, self.dictionary, self.window_size, self.metric, self.topn)
        return cm.get_coherence()

    def set_model(self, model):
        self.model = model
        if self.logger=='visdom':
            self.viz_window = Visdom()
            self.visualize = Visualize(self.viz_env, self.viz_window)
        if self.logger=='shell':
            self.log = Log(type(self.model))

    def on_epoch_end(self, epoch, topics=None):
        # provide topics after current epoch if coherence is not valid for this topic model
        value = self.get_metric(topics=topics)
        if self.logger=='visdom':
            self.visualize.line_plot(value, epoch, 'Epochs', 'Coherence', 'Coherence (%s)' % self.metric)
        if self.logger=='shell':
            statement = "Coherence estimate: %.3f" % value
            self.log.log_metric(statement)


class Visualize(object):
    def __init__(self, viz_env, viz_window):
        self.env = viz_env
        self.window = viz_window

    def line_plot(self, value, epoch, xlabel, ylabel, title):
        value = np.array([value])
        epoch = np.array([epoch])
        if epoch == 0:
            self.viz_metric = self.window.line(Y=value, X=epoch, env=self.env, opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
        else:
            self.window.updateTrace(Y=value, X=epoch, env=self.env, win=self.viz_metric)

    # def heatmap_plot():


class Log(object):
    def __init__(self, model_type):
        model_type = str(model_type)
        self.log_type = logging.getLogger(model_type)
        
    def log_metric(self, statement):
            self.log_type.info(statement)

