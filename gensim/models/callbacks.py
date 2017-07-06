import gensim
import logging
import numpy as np
from visdom import Visdom


class Coherence(object):
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

    def get_value(self, topics=None):
        if topics is None:
            topics = self.topics
        cm = gensim.models.CoherenceModel(self.model, topics, self.texts, self.corpus, self.dictionary, self.window_size, self.metric, self.topn)
        return cm.get_coherence()

class Perplexity(object):
	def __init__(self, arg):
		self.arg = arg

	def get_value(self, arg):

		
class Diff(object):
	def __init__():

	def get_value(self):


class Convergence(object):
	def __init__(self, arg):
		self.arg = arg

	def get_value(self):
		

class Callback(object):
	def __init__(self, metrics=None):
		# list of metric callbacks
		self.metrics = metrics

    def set_model(self, model):
        self.model = model
        if self.logger=='visdom':
            self.viz = Visdom()
            self.windows = []
            self.diff_mat = []
        if self.logger=='shell':
        	model_type = str(model_type)
            self.log_type = logging.getLogger(model_type)

    def on_epoch_end(self, epoch, topics=None):
        # provide topics after current epoch if coherence is not valid for this topic model
        for i, metric in enumerate(self.metrics):
	        value = metric.get_value(topics=topics)

		    if metric.logger=="visdom":
			    if epoch==0:
			    	if type(value)==int:
				        viz_metric = self.viz.line(Y=np.array([value]), X=np.array([epoch]), env=metric.env, opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
				        self.windows.append(copy.deepcopy(viz_metric))
				    else:
				    	self.diff_mat = np.array([value])
				    	viz_metric = self.viz.heatmap(X=np.array(self.diff_mat).T, env=metric.env, opts=dict(xlabel='Epochs', ylabel='Topic', title='Diff (%s)' % diff_distance))
			    		self.windows.append(copy.deepcopy(viz_metric))
			    else:
			    	if type(value)==int:
			    		self.viz.updateTrace(Y=np.array([value]), X=np.array([epoch]), env=metric.env, win=self.windows[i])
			    	else:
			    		self.diff_mat = np.concatenate((self.diff_mat, np.array([value])))
			    		self.viz.heatmap(X=np.array(self.diff_mat).T, env=viz_env, win=self.viz_diff, opts=dict(xlabel='Epochs', ylabel='Topic', title='Diff (%s)' % diff_distance))

	        if metric.logger=='shell':
	            statement = "Estimate: %.3f" % value
	            self.log_type.info(statement)


