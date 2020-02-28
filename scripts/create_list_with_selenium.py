#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:19:15 2020
"""

# %%
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver import Firefox
import time

# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
# import os

# %%
options = Options()
options.headless = True

profile = FirefoxProfile()
profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/octet-stream')
# profile.set_preference('browser.downloadDir', '/home/andyb/deeply_thinking_potato/data/potts/')
browser = Firefox(firefox_profile=profile, options=options)

# %%
url_template = 'https://byu.app.box.com/v/ProteinStructurePrediction/folder/87136020056?page={page}'
file = open('links.tsv', 'w')

# %%
for page in range(1, 1428):

    url = url_template.format(page=page)
    browser.get(url)
    time.sleep(5)

    links = browser.find_elements_by_xpath("//*[contains(@class, 'item-link')]")
    n_links = len(links)
    print(f'{page}\t{len(links)}')

    for link in links:
        href = link.get_attribute('href')
        name = link.get_attribute('text')
        print(f'{name}\t{href}\t{page}\t{n_links}', file=file, flush=True)
        # print(f'{name}\t{href}\t{page}\t{n_links}')

# %%
browser.close()
