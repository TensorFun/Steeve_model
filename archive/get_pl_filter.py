# coding: utf-8

import json, os

def request_web():
    from collections import defaultdict
    from bs4 import BeautifulSoup
    import requests
    
    res = requests.get('https://www.wappalyzer.com/datasets').content
    html = BeautifulSoup(res, 'lxml')
    
    pl_filter = defaultdict(lambda: [])
    categories = html.find_all("div", {"class": "category"})
    for category in categories:
        labels = category.find_all("label")
        for label in labels:
            if label.has_attr("data-slug"):
                pl_filter[category["data-slug"]].append(label.text.strip())
                
    with open('pl_filter.json', 'w', encoding='utf8') as fs:
        fs.write(json.dumps(pl_filter))
        
    return pl_filter

def get_pl_filter():
    if os.path.isfile('pl_filter.json'):
        text = open('pl_filter.json', 'r', encoding='utf8').read()
        return json.loads(text)
    else:
        return request_web()
    
if __name__ == '__main__':
    pl_filter = request_web()



