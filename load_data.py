import json
import gzip as gz


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


def load_files(file_names=None, max_lines=None):
    """
    :param file_names: List of files paths to load
    :param max_lines: Max number of lines to return
    :return text: List of text, one string per ad
    :return labels: List of labels, one per ad
    :return indices: List of indices, one per ad
    :return ad_id: List of ad ids, one per ad
    :return phone: List of tuples, each tuple contains strings of each phone number in ad
    """
    if file_names is None:
        file_names = ['data/ht_training_UPDATED.gz', 'data/ht_training_2.gz']
    text, labels, ad_id, phone = zip(*(d for d in _extract_data(file_names, max_lines=max_lines)))
    return text, labels, ad_id, phone


def _extract_data(filenames, max_lines=None):
    """
    Extracts ad text, id, and label (0 or 1)s
    :param filenames: gz files containing json objects
    """
    count = 0
    for file_name in filenames:
        if count == max_lines:
            break
        with gz.open(file_name, 'r') as f:
            for line in f:
                d = json.loads(line)
                try:
                    if 'extracted_text' in d['ad']:
                        text = d['ad']['extracted_text']
                    else:
                        text = d['ad']['extractions']['text']['results'][0]
                    if 'class' in d:
                        if d['class'] == 'positive':
                            yield text.encode('utf8'), 1, d['ad']['_id'], tuple(d['phone'])
                        else:
                            yield text.encode('utf8'), 0, d['ad']['_id'], tuple(d['phone'])
                    else:
                        yield text.encode('utf8'), None, d['ad']['_id'], tuple(d['phone'])
                    count += 1
                except:
                    print d
                if count == max_lines:
                    break
