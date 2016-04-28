import json
import string
from options import space
from hyperopt import fmin, tpe, Trials, space_eval
from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score
from hyperopt import STATUS_OK
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from load_data import load_files
from string import maketrans

class Featurizer(object):
    def __init__(self,args=None):
        self.COUNT_BASE = {
            'strip_accents': None,
            'stop_words': 'english',
            'ngram_range': (1, 1),
            'analyzer': 'word',
            'max_df': 1.0,
            'min_df': 1,
            'max_features': None,
            'binary': False
            }
        #for unicode:
        #self.remove_punct_map = dict((ord(char), u' ') for char in string.punctuation)
        #self.remove_digit_map = dict((ord(char), u' ') for char in string.digits)
        #for string:
        self.remove_punct_map = maketrans(string.punctuation,' '*len(string.punctuation))
        self.remove_digit_map = maketrans(string.digits,' '*len(string.digits))
        self.printable = frozenset(string.printable)
        self.strip_digits=False
        self.strip_punct=False
        self.ascii_only = False
        self.vec = None
        if not args is None:
            feat_options = args['features']
            for key,value in self.COUNT_BASE.items():
                if key not in feat_options:
                    feat_options[key] = value
            self.options = feat_options
            self.strip_digits=args['cleaning']['strip_digits']
            self.strip_punct=args['cleaning']['strip_punct']
        else:
            self.options = self.COUNT_BASE

    def strip_non_ascii(self, text):
        return filter(lambda x: x in self.printable, text)
    
    @staticmethod
    def clean_text(text, rm_map):
        """
        :param text: String
        :param rm_map: Dictionary of characters to replace [ordinal of character to replace, unicode character to replace with]
        :return: Cleaned string
        """
        return text.translate(rm_map)
    
    def get_features(self,data, options):
        #if this operation is very expensive then we should store the results
        self.vec = CountVectorizer(strip_accents=options['strip_accents'], stop_words=options['stop_words'],
                              ngram_range=options['ngram_range'], analyzer=options['analyzer'],
                                max_df=options['max_df'], min_df=options['min_df'],
                              max_features=options['max_features'], binary = options['binary'])
        return self.vec.fit_transform(data)

    def set_options(self,args):
        feat_options = args['features']
        for key,value in self.COUNT_BASE.items():
            if key not in feat_options:
                feat_options[key] = value
        self.options = feat_options
        self.strip_digits=args['cleaning']['strip_digits']
        self.strip_punct=args['cleaning']['strip_punct']

    def set_ascii(self, value):
        if value:
            self.ascii_only = True

    def run(self,text):
        #walk through text cleaning options
        if self.ascii_only:
            text = [self.strip_non_ascii(t) for t in text]
        if self.strip_digits:
            text = [self.clean_text(t, self.remove_digit_map) for t in text]
        if self.strip_punct:
            text = [self.clean_text(t, self.remove_punct_map) for t in text]

        return self.get_features(text,self.options)

    def run_testdata(self,text):
        if self.ascii_only:
            text = [self.strip_non_ascii(t) for t in text]
        if self.strip_digits:
            text = [self.clean_text(t, self.remove_digit_map) for t in text]
        if self.strip_punct:
            text = [self.clean_text(t, self.remove_punct_map) for t in text]

        return self.vec.transform(text)


class ClassifierLearner(object):
    def __init__(self, labels, text, num_folds=5,folds=None,cjobs=2,cvjobs=10):
        #process and prepare text
        self.cjobs=cjobs#n_jobs parameter for classifier
        self.cvjobs = cvjobs #n_jobs parameter for cross validation
        self.Featurizer = Featurizer()
        non_ascii = [self.Featurizer.strip_non_ascii(t) for t in text]
        self.text = {'raw':text,'ascii':non_ascii}
        #store labels
        self.labels = labels
        #set up CV
        if not folds is None:
            self.skf = folds
        else:
            self.skf = StratifiedKFold(labels, num_folds, random_state=137)
       

    def call_experiment(self, args):
        """
        :param args: Hyperopt parameters
        :return: Hyperopt feedback
        """
        #set up classifier
        model = self.get_model(args)   

        #get raw or non-ascii text (removing non-ascii is expensive)      
        text = self.text[args['text']['text']]#get raw text or ascii only text

        self.Featurizer.set_options(args) 

        X = self.Featurizer.run(text)

        f1 = cross_val_score(model, X, self.labels, cv=self.skf, scoring='f1', n_jobs=self.cvjobs).mean()

        print f1
        return {'loss': -f1, 'status': STATUS_OK}    

    def get_model(self,args):
        if args['model']['model'] == 'LR':
            model = lr(penalty=args['model']['regularizer_lr'], C=args['model']['C_lr'],n_jobs=self.cjobs)
        elif args['model']['model'] == 'SVM':
            if args['model']['regularizer_svm'] == 'l1':
                #squared hinge loss not available when penalty is l1. 
                model = svm.LinearSVC(C=args['model']['C_svm'], penalty=args['model']['regularizer_svm'],dual=False,n_jobs=self.cjobs)#loss='hinge')
            else:
                model = svm.LinearSVC(C=args['model']['C_svm'], penalty=args['model']['regularizer_svm'],n_jobs=self.cjobs)
        return model

    def run(self,max_evals=100):
        trials = Trials()
        best = fmin(self.call_experiment,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)
        #print best
        args = space_eval(space, best)
        #print "losses:", [-l for l in trials.losses()]
        #print max([-l for l in trials.losses()])
        
        #TODO
        #return model and featurizer. Don't forget to set ascii option.
        featurizer = Featurizer(args)
        if args['text']['text'] == 'ascii':
            featurizer.set_ascii(True)
        return self.get_model(args),featurizer 

if __name__ == '__main__':
    text, labels, ad_id, phone = load_files()
    CL = ClassifierLearner(labels, text, num_folds=5)
    mode, featurizer = CL.run(max_evals=5)
