#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 12:43:32 2019

@author: yq

REFERENCE:
    https://towardsdatascience.com/how-to-web-scrape-with-python-in-4-minutes-bc49186a8460
"""

import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import os

#years = [2002, 2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017]
#
#for year in years:
#    os.mkdir("yamaha/{}".format(year))
#    
#    url = "http://www.piano-e-competition.com/midi_{}.asp".format(year)
#    response = requests.get(url)
#    soup = BeautifulSoup(response.text, "html.parser")
#    a_tags = soup.findAll("a")
#    
#    for a in range(len(a_tags)):
#        try:
#            if ".MID" in a_tags[a]["href"] or ".mid" in a_tags[a]["href"]:
#                download = "http://www.piano-e-competition.com" + a_tags[a]["href"]
#                b = os.path.split(a_tags[a]["href"])[-1]
#                save_path = os.path.join("yamaha/{}/".format(year), b)
#                urllib.request.urlretrieve(download, save_path)
#                
#                print("[INFO] Downloaded {}".format(download))
#                time.sleep(1)
#        except:
#            continue
#    
#
os.mkdir("dataset/chinese")
url = "http://ingeb.org/catcn.html"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
a_tags = soup.findAll("a")

for a in range(len(a_tags)):
    try:
        if ".MID" in a_tags[a]["href"] or ".mid" in a_tags[a]["href"]:
            download = a_tags[a]["href"]
            download = "http://ingeb.org/" + download
            b = os.path.split(a_tags[a]["href"])[-1]
            save_path = os.path.join("dataset/chinese/", b)
            urllib.request.urlretrieve(download, save_path)
            
            print("[INFO] Downloaded {}".format(download))
            time.sleep(1)
    except:
        continue