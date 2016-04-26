import json
from itertools import izip


class TextCleaner(object):
    def __init__(self):
        characters_to_replace = ',.&!+:;?"#()\'*+,./<=>@[\\]^_`{|}~\n'
        self.remove_white_map = dict((ord(char), u' ') for char in characters_to_replace)

    def clean_text(self, text):
        """
        Replaces some characters with white space
        :param text: String
        :return: Text with chars_to_replace replaced with white space
        """
        return text.translate(self.remove_white_map)


def load_files(file_names=None):
    """
    :param file_names: List of files paths to load
    :return text: List of text
    :return labels: List of labels
    :return indices: List of indices
    :return ad_id: List of ad ids
    """
    if file_names is None:
        file_names = ['data/ht_training_UPDATED', 'data/ht_training_2']
    data = []
    for file_name in file_names:
        with open(file_name, 'r') as f:
            data+=[json.loads(line) for line in f]
    text, labels, indices, ad_id,phone = zip(*(d for d in _extract_data(data)))
    #return dictionaries of ad_id:text,ad_id:label,ad_id:phone
    text_dict = {key:value for key,value in izip(ad_id,text)}
    label_dict = {key:value for key,value in izip(ad_id,labels)}
    phone_dict = {key:value for key,value in izip(ad_id,phone)}
    return text_dict,label_dict,phone_dict


def _extract_data(data):
    """
    Extracts ad text, id, and label (0 or 1)s
    :param data: JSON object
    """
    for i, d in enumerate(data):
        try:
            if 'extracted_text' in d['ad']:
                text = d['ad']['extracted_text']
            else:
                text = d['ad']['extractions']['text']['results'][0]    
            if d['class'] == 'positive':
                yield text, 1, i, d['ad']['_id'],tuple(d['phone'])
            else:
                yield text, 0, i, d['ad']['_id'],tuple(d['phone'])
        except:
            print d
