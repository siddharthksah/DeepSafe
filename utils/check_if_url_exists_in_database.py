# Python 3
from urllib.parse import urlparse, parse_qsl, unquote_plus

class Url(object):
    '''A url object that can be compared with other url orbjects
    without regard to the vagaries of encoding, escaping, and ordering
    of parameters in query strings.'''

    def __init__(self, url):
        parts = urlparse(url)
        _query = frozenset(parse_qsl(parts.query))
        _path = unquote_plus(parts.path)
        parts = parts._replace(query=_query, path=_path)
        self.parts = parts

    def __eq__(self, other):
        return self.parts == other.parts

    def __hash__(self):
        return hash(self.parts)

def check_url_for_deepfake_in_database(url1, url2):
    if Url(url1) == Url(url2):
        print("yay")

url1 = "https://mlops.githubapp.com/"
url2 = "https://mlops.githubapp.com/"

check_url_for_deepfake_in_database(url1, url2)