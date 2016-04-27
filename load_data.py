import json
import gzip as gz
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
        file_names = ['data/ht_training_UPDATED.gz', 'data/ht_training_2.gz']
    text, labels, ad_id, phone = zip(*(d for d in _extract_data(file_names)))
    return text, labels, ad_id, phone


def _extract_data(filenames):
    """
    Extracts ad text, id, and label (0 or 1)s
    :param filenames: gz files containing json objects
    """
    for file_name in filenames:
        with gz.open(file_name,'r') as f:
            for line in f:
                d = json.loads(line)
                try:
                    if 'extracted_text' in d['ad']:
                        text = d['ad']['extracted_text']
                    else:
                        text = d['ad']['extractions']['text']['results'][0]    
                    if d['class'] == 'positive':
                        yield text, 1, d['ad']['_id'],tuple(d['phone'])
                    else:
                        yield text, 0, d['ad']['_id'],tuple(d['phone'])
                except:
                    print d
