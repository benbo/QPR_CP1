import json
import string

#global variables
charsWHITE =',.&!+:;?"#()\'*+,./<=>@[\\]^_`{|}~\n'
remove_white_map = dict((ord(char), u' ') for char in charsWHITE)

#functions

def clean_text(t):
    return t.translate(remove_white_map)

def extract_data(data):
    for i,d in enumerate(data):
        try:
            if 'extracted_text' in d['ad']:
                text = d['ad']['extracted_text']
            else:
                text = d['ad']['extractions']['text']['results'][0]    
            if d['class'] == 'positive':
                yield clean_text(text),1,i,data[0]['ad']['_id']
            else:
                yield clean_text(text),0,i,data[0]['ad']['_id']
        except:
            print d

def write_text(text,out = 'ht_training_text.txt'):
    with open(out,'w') as f:
        for line in text:
            f.write(line.encode('utf-8'))
            f.write('\n')
#load data
filename = 'ht_training_UPDATED'
with open(filename,'r') as f:
    data =[json.loads(line) for line in f] 

filename = 'ht_training_2_2'
with open(filename,'r') as f:
    data = data + [json.loads(line) for line in f] 

text,labels,indices,ad_id = zip(*[ d for d in extract_data(data)])
