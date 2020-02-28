# based on https://realpython.com/modern-web-automation-with-python-and-selenium/

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
def get_elements(browser):
    wait = WebDriverWait(browser, 60)

    next_button = wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//*[contains(@class, 'bcpr-navigate-right')]")
            )
        )

    download_button = wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//*[contains(@class, 'bp-btn-primary')]")
            )
        )

    filename = wait.until(
        EC.presence_of_element_located(
            (By.XPATH, "//*[contains(@class, 'shared-folder-preview-file-name')]")
            )
        ).text

    url = browser.current_url

    return (url, filename, download_button, next_button)


# %%
options = Options()
options.headless = False

profile = FirefoxProfile()
profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/octet-stream')
# profile.set_preference('browser.downloadDir', '/home/andyb/deeply_thinking_potato/data/potts/')
browser = Firefox(firefox_profile=profile, options=options)

# %%
# url = 'https://byu.app.box.com/v/ProteinStructurePrediction/file/525413022614'
# url = 'https://byu.app.box.com/v/ProteinStructurePrediction/file/525406802922'
# url = 'https://byu.app.box.com/v/ProteinStructurePrediction/file/525406139424'
# url = 'https://byu.app.box.com/v/ProteinStructurePrediction/file/525418936666'
url = 'https://byu.app.box.com/v/ProteinStructurePrediction/file/525402645504'

# https://files.physics.byu.edu/data/prospr/potts/

filename = None
n_errors = 0
# file = open('downloaded.log', 'w')

browser.get(url)
while True:
    try:
        state = 'done'
        url, filename, download_button, next_button = get_elements(browser)
        if not os.path.exists(f'/home/andyb/Downloads/{filename}'):
            download_button.click()
            while not os.path.exists(f'/home/andyb/Downloads/{filename}.part'):
                time.sleep(5)
            state = 'downloading'
        # print(f'{url}\t{filename}\t{state}', file=file, flush=True)
        print(f'{url}\t{filename}\t{state}')
        next_button.click()
        time.sleep(5)
        n_errors = 0
    except TimeoutException:
        n_errors += 1
        print(f'TimeoutException({n_errors}) - refreshing page {url}')
        browser.get(url)
    except StaleElementReferenceException:
        n_errors += 1
        print(f'StaleElementReferenceException({n_errors}) - refreshing page {url}')
        browser.get(url)

    if n_errors > 4:
        break

input("Press enter to exit the program")
# %%
browser.close()
