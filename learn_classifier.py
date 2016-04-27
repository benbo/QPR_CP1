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
        self.remove_punct_map = dict((ord(char), u' ') for char in string.punctuation)
        self.remove_digit_map = dict((ord(char), u' ') for char in string.digits)
        self.printable = frozenset(string.printable)
        self.strip_digits=False
        self.strip_punct=False
        self.ascii_only = False
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
    
    @staticmethod
    def get_features(data, options):
        #if this operation is very expensive then we should store the results
        vec = CountVectorizer(strip_accents=options['strip_accents'], stop_words=options['stop_words'],
                              ngram_range=options['ngram_range'], analyzer=options['analyzer'],
                                max_df=options['max_df'], min_df=options['min_df'],
                              max_features=options['max_features'], binary = options['binary'])
        return vec.fit_transform(data)

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



class ClassifierLearner(object):
    def __init__(self, label_dict, text_dict, num_folds=5):
        #process and prepare text
        self.Featurizer = Featurizer()
        keys = text_dict.keys()
        raw_text = [text_dict[k] for k in keys]
        non_ascii = [self.Featurizer.strip_non_ascii(t) for t in raw_text]
        self.text = {'raw':raw_text,'ascii':non_ascii}
        #store labels
        self.labels = [label_dict[k] for k in keys]
        #set up CV
        self.skf = StratifiedKFold(self.labels, num_folds, random_state=137)

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

        f1 = cross_val_score(model, X, self.labels, cv=self.skf, scoring='f1', n_jobs=8).mean()

        print f1
        return {'loss': -f1, 'status': STATUS_OK}    

    @staticmethod
    def get_model(args):
        if args['model']['model'] == 'LR':
            model = lr(penalty=args['model']['regularizer_lr'], C=args['model']['C_lr'])
        elif args['model']['model'] == 'SVM':
            if args['model']['regularizer_svm'] == 'l1':
                #squared hinge loss not available when penalty is l1. 
                model = svm.LinearSVC(C=args['model']['C_svm'], penalty=args['model']['regularizer_svm'],dual=False)#loss='hinge')
            else:
                model = svm.LinearSVC(C=args['model']['C_svm'], penalty=args['model']['regularizer_svm'])
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
    #text_dict,label_dict,phone_dict = load_files(['data/ht_training_UPDATED.gz'])
    text_dict,label_dict,phone_dict = load_files()
    CL = ClassifierLearner(label_dict, text_dict, num_folds=5)
    CL.run(max_evals=5)
