import json


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
            for line in f:
                data.append(json.loads(line))
    text, labels, indices, ad_id = zip(*[d for d in _extract_data(data)])
    return text, labels, indices, ad_id


def _extract_data(data):
    """
    Extracts ad text, id, and label (0 or 1)s
    :param data: JSON object
    """
    cleaner = TextCleaner()
    for i, d in enumerate(data):
        try:
            if 'extracted_text' in d['ad']:
                text = d['ad']['extracted_text']
            else:
                text = d['ad']['extractions']['text']['results'][0]    
            if d['class'] == 'positive':
                yield cleaner.clean_text(text), 1, i, data[0]['ad']['_id']
            else:
                yield cleaner.clean_text(text), 0, i, data[0]['ad']['_id']
        except:
            print d