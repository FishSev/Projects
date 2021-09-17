from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import ctypes #для диалогового окна
import time # для sleep

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


chromedriver = 'C:/_Work/Python/Automat/chromedriver'
options = webdriver.ChromeOptions()
#options.add_argument('headless')  # для открытия headless-браузера
browser = webdriver.Chrome(executable_path=chromedriver, chrome_options=options)
browser.maximize_window()

browser.get('https://tradedesk.dmp.kz/Sk/DMP/')

browser.find_element_by_xpath("//nav/a[1]").click()

LoginTXT="techstack.dmp.kz@test.test"
PasswordTXT="1jhnlIJN!k#"

browser.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)


time.sleep(1)

try:
    EmailElem = browser.find_element_by_name('mail')
    ExcMail=""
except Exception:
    ExcMail=Exception
    print("Problem with e-mail element")
if ExcMail=="":
    EmailElem.send_keys(LoginTXT)

try:
    PasswordElem = browser.find_element_by_name('password')
    ExcPass=""
except Exception:
    ExcPass=Exception
    print("Problem with password element")
if ExcPass=="":
    PasswordElem.send_keys(PasswordTXT)

browser.find_element_by_tag_name('button').click()

Err="-"
MyTime=0
while Err!="":
    try:
        ClassElem = browser.find_element_by_class_name('metro')
        Err=""
    except Exception:
        Err=Exception
        time.sleep(1)

browser.get('https://techstack.dmp.kz/#/products/')


# ctypes.windll.user32.MessageBoxW(0, browser.find_element_by_xpath("//html").text, "Your title", 1)

# if browser.find_element_by_xpath("/button").text == "Запомнить меня":
#     ctypes.windll.user32.MessageBoxW(0, "here", "Your title", 1)

#browser.execute_script("return document.readyState")





# time.sleep(50)

# browser.quit()