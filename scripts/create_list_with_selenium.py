#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:19:15 2020
"""

# %%
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import time
import os

# %%
options = Options()
options.headless = False

profile = FirefoxProfile()
profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/octet-stream')
# profile.set_preference('browser.downloadDir', '/home/andyb/deeply_thinking_potato/data/potts/')
browser = Firefox(firefox_profile=profile, options=options)

# %%
url_template = 'https://byu.app.box.com/v/ProteinStructurePrediction/folder/87136020056/page={page}'

# %%
for page in range(1, 1428):

    url = url_template.format(page=page)
    browser.get(url)
    time.sleep(10)

    links = browser.find_elements_by_xpath("//*[contains(@class, 'item-link')]")
    print(f'{page}\t{len(links)}')

# %%
links[0].get_attribute('href')
links[0].get_attribute('text')



