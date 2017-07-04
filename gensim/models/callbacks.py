TopicMetrics(object):
    
    def get_coherence(self, model, corpus, coherence, texts, window_size, topn):
        corpus_words = sum(cnt for document in chunk for _, cnt in document)
        cm = gensim.models.CoherenceModel(model=self, corpus=corpus, texts=texts, coherence=coherence, window_size=coherence_window_size, topn=coherence_topn)
        return cm.get_coherence(), corpus_words

    def get_perplexity(self, model, corpus):
        corpus_words = sum(cnt for document in corpus for _, cnt in document)
        perwordbound = model.bound(corpus) / corpus_words
        return np.exp2(-perwordbound), corpus_words

    def get_diff(self, model, corpus, previous):
        diff_matrix = self.diff(previous, distance=diff_distance)[0]
        diff_diagonal = np.diagonal(diff_matrix)
        return diff_diagonal

    def get_convergence(self, model, corpus, previous):
        diff_matrix = self.diff(previous, distance=diff_distance)[0]
        diff_diagonal = np.diagonal(diff_matrix)
        convergence = np.sum(diff_diagonal)
        return convergence


Log(TopicMetrics):

    def __init__(self, coherence="u_mass", coherence_texts=None, coherence_window_size=None, coherence_topn=10, diff_distance="kullback_leibler"):

    def set_model(self, model):
        self.model = model
        if diff_distance:
            previous = copy.deepcopy(model)

    def on_epoch_end(self, epoch, chunk):

        if coherence:
            cm, corpus_words = self.get_coherence(self.model, self.corpus)
            logger.info("%.3f coherence estimate based on a held-out corpus of %i documents with %i words", cm, len(chunk), corpus_words)

        if perplexity:
            pl, corpus_words = self.get_perplexity(self.model, self.corpus)
            logger.info("%.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words" % (perwordbound, pl, len(chunk), corpus_words))

        if diff:
            df = self.get_diff()
            prev_epoch = epoch-1
            logger.info("Topic difference between %s and %s epoch: %s", prev_epoch, epoch, df)

        if convergence:
            cv = self.get_convergence()
            logger.info("Convergence at %s epoch: ", cv)



Visualizer(TopicMetrics):

    def __init__(self, viz_env=None, coherence="u_mass", coherence_texts=None, coherence_window_size=None, coherence_topn=10, diff_distance="kullback_leibler"):

    def set_model(self, model, corpus):
        self.viz_window = Visdom()
        self.model = model
        self.corpus = corpus
        if diff_distance:
            previous = copy.deepcopy(model)

    def on_epoch_end(self, epoch):

        if coherence:
            # calculate coherence
            cm = self.get_coherence(self.model, self.corpus)
            coherence_value = np.array([cm])
            if epoch == 0:
                # initial plot window
                self.viz_coherence = self.viz_window.line(Y=coherence_value, X=np.array([epoch]), env=viz_env, opts=dict(xlabel='Epochs', ylabel='Coherence', title='Coherence (%s)' % coherence))
            else:
                self.viz_window.updateTrace(Y=coherence_value, X=np.array([epoch]), env=viz_env, win=self.viz_coherence)

        if perplexity:
            # calculate perplexity
            pl = self.get_perplexity(self.model, self.corpus)
            perplexity_value = np.array([pl])
            if epoch == 0:
                # initial plot window
                self.viz_perplexity = self.viz_window.line(Y=perplexity, X=np.array([epoch]), env=viz_env, opts=dict(xlabel='Epochs', ylabel='Perplexity', title='Perplexity'))
            else:
                self.viz_window.updateTrace(Y=perplexity, X=np.array([epoch]), env=viz_env, win=self.viz_perplexity)

        if diff:
            # calculate diff
            df = self.get_diff()
            diff_value = np.array([df])
            if epoch == 0:
                # initial plot window
                self.diff_mat = np.array([diff_diagonal])
                self.viz_diff = self.viz_window.heatmap(X=np.array(self.diff_mat).T, env=viz_env, opts=dict(xlabel='Epochs', ylabel='Topic', title='Diff (%s)' % diff_distance)) 
            else:
                # update the plot with each epoch
                self.diff_mat = np.concatenate((self.diff_mat, np.array([diff_diagonal])))
                self.viz_window.heatmap(X=np.array(self.diff_mat).T, env=viz_env, win=self.viz_diff, opts=dict(xlabel='Epochs', ylabel='Topic', title='Diff (%s)' % diff_distance))

        if convergence:
            # calculate convergence
            cm = self.get_convergence()
            coherence_value = np.array([cm])
            if epoch == 0:
                # initial plot window
                self.viz_convergence = viz_window.line(Y=convergence, X=np.array([epoch]), env=viz_env, opts=dict(xlabel='Epochs', ylabel='Convergence', title='Convergence (%s)' % diff_distance))
            else:
                viz_window.updateTrace(Y=convergence, X=np.array([epoch]), env=viz_env, win=self.viz_convergence)

