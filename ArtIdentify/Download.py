'''
Created on Dec 2, 2021

@author: starw
'''
import os
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup, SoupStrainer
import requests
import json
from os.path  import basename

# This file is used for downloading the data to local and creating a 
# format to Toloka that makes up the task pool

def main():
    url = "https://github.com/wangfanli12/Edvard_Munch/tree/main/Edvard_Munch"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, parse_only=SoupStrainer('a'))
    page = urllib.request.urlopen(url)
    linklist = []
    for link in soup:
        if link.has_attr('href'):
            if "/wangfanli12/Edvard_Munch/blob/main/Edvard_Munch/" in link['href']:
                linklist.append(link['href'])
                
    database = []
    i = 0
    for l in linklist:
        data = {}
        path = {}
        data['id'] = str(i)
        i = i + 1
        rawImage = "https://raw.githubusercontent.com" + l.replace('blob/','')
        print(rawImage)
        path['image'] = rawImage
        
        data['input_values'] = path
        database.append(data)
        
        
    python_file = open("data.json", "w")

    json.dump(database, python_file, ensure_ascii=False, indent=4)
    python_file.close()
    
if __name__ == '__main__':
    main()
