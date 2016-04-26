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


class ClassifierLearner(object):
    def __init__(self,labels,text_dict):

        self.COUNT_BASE = {
            'strip_accents':None, 'stop_words':'english', 'ngram_range':(1, 1), 'analyzer':'word', 'max_df':1.0, 'min_df':1, 'max_features':None, 'binary':False
            }

        self.remove_punct_map = dict((ord(char), u' ') for char in string.punctuation)
        self.remove_digit_map = dict((ord(char), u' ') for char in string.digits)
        self.printable = frozenset(string.printable)
        self.skf = StratifiedKFold(labels, 5,random_state=17)
        self.labels=labels
        self.text=text_dict
    def clean_text(self,t,rm_map):
        return t.translate(rm_map)

    def strip_non_ascii(self,t):
        return filter(lambda x: x in self.printable, t) 

    def get_features(self,data,options):
        #if this operation is very expensive then we should store the results
        vec = CountVectorizer(strip_accents=options['strip_accents'], stop_words=options['stop_words'], ngram_range=options['ngram_range'], analyzer=options['analyzer'],
              max_df=options['max_df'], min_df=options['min_df'], max_features=options['max_features'], binary = options['binary'])
        return vec.fit_transform(data)

    def call_experiment(self,args):
        #set up classifier
        model = args['model']    
        if model['model'] == 'LR':
            model = lr(penalty=model['regularizer_lr'], C=model['C_lr'])
        elif model['model'] == 'SVM':
            if model['regularizer_svm'] == 'l1':
                #squared hinge loss not available when penalty is l1. 
                model = svm.LinearSVC(C=model['C_svm'], penalty=model['regularizer_svm'],dual=False)#loss='hinge')
            else:
                model = svm.LinearSVC(C=model['C_svm'], penalty=model['regularizer_svm'])
      
        #walk through text cleaning options
        text = self.text[args['text']['text']]#get raw text or ascii only text

        if args['cleaning']['strip_digits']:
            text = [self.clean_text(t,self.remove_digit_map) for t in text]    
        if args['cleaning']['strip_punct']:
            text = [self.clean_text(t,self.remove_punct_map) for t in text]    

     
        feat_options = args['features']
        #set up features
        for key,value in self.COUNT_BASE.items():
            if key not in feat_options:
                feat_options[key] = value
        X = self.get_features(text,feat_options)

        f1 = cross_val_score(model, X, self.labels, cv=self.skf,scoring='f1',n_jobs=8).mean()

        print f1
        return {'loss': -f1, 'status': STATUS_OK}    

if __name__ == '__main__':
    raw_text,labels,indices,ad_ids,phones = load_files()
    ascii_text = [strip_non_ascii(t) for t in raw_text]
    CL = ClassifierLearner(labels,{'raw':raw_text,'ascii':ascii_text})
    trials = Trials()
    best = fmin(CL.call_experiment,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)
    print best
    print space_eval(space, best)
    print "losses:", [-l for l in trials.losses()]
    print max([-l for l in trials.losses()])
