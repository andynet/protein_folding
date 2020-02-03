# based on https://realpython.com/modern-web-automation-with-python-and-selenium/

# %%
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver import Firefox
import time

# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.alert import Alert
# %%
options = Options()
options.headless = True

profile = FirefoxProfile()
profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/octet-stream')
# profile.set_preference('browser.downloadDir', '/home/andyb/deeply_thinking_potato/data/potts/')
browser = Firefox(firefox_profile=profile, options=options)

# %%
browser.get('https://byu.app.box.com/v/ProteinStructurePrediction/file/525413022614')

while True:
    print(browser.current_url)
    while (len(elements := browser.find_elements_by_tag_name('button')) < 40):
        time.sleep(5)

    for element in elements[::-1]:
        if element.text == 'Download':
            break

    element.click()

    elements = browser.find_elements_by_xpath('//*[@title="Next File"]')
    if len(elements) < 1:
        break
    else:
        elements[0].click()

browser.close()
