from hyperopt import hp

space = {
    #'model': hp.choice('model', [
    #    {
    #        'model': 'SVM',
    #        'regularizer_svm': hp.choice('regularizer_svm', ['l1', 'l2']),
    #        'C_svm': hp.loguniform('C_svm', -1.15, 9.2)
    #    },
    #    {
    #        'model': 'LR',
    #        'regularizer_lr': hp.choice('regularizer_lr', ['l1', 'l2']),
    #        'C_lr': hp.loguniform('C_lr', -1.15, 9.2)
    #    }
    #]),
    'model':
        {
            'model': 'LR',
            'regularizer_lr': hp.choice('regularizer_lr', ['l1', 'l2']),
            'C_lr': hp.loguniform('C_lr', -1.15, 9.2)
        },
'features' : {
    'strip_accents':hp.choice('strip_accents', ['ascii', 'unicode', None]),
    'ngram_range': hp.choice('ngram_range', [(1,1), (1,2), (2,2)]),
    'analyzer':hp.choice('analyzer',['word','char']),
    'max_df':hp.choice('max_df', [0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0]),#'max_df':hp.uniform('max_df', 0.7,1.0)
    'min_df':hp.choice('min_df', [0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.1,1]),
    'binary':hp.choice('binary', [True,False])
},
'text':hp.choice('text', [
        {
            'text': 'raw',
        },
        {
            'text': 'ascii',
        }
    ]),
'cleaning':{
      'strip_digits':hp.choice('strip_digits', [True,False]),
      'strip_punct':hp.choice('strip_punct', [True,False]),
    }
}
