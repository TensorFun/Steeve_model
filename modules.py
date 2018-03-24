
# coding: utf-8

# In[10]:

# Extract PL

from flashtext import KeywordProcessor

pls = open('Rule.txt', 'r', encoding='utf8').read().split('\n')

keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(pls)

def get_pl_keywords(content):
    return keyword_processor.extract_keywords(content)

def norm_pls(pls):
    '''
    Params: ['pl_A', 'pl_B', 'pl_C']
    
    Returns: list for normalized pls
    '''
    return [pl.lower().replace(' ', '_').replace('.js', '') for pl in pls]

