
# coding: utf-8

# In[ ]:

from flashtext import KeywordProcessor

pls = open('Rule.txt', 'r', encoding='utf8').read().split('\n')

keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(pls)


# In[ ]:

def get_pl_keywords(content):
    return keyword_processor.extract_keywords(content)


# In[ ]:




# In[ ]:



