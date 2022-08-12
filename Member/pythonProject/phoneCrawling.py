from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import os, time, random
from bs4 import BeautifulSoup
import urllib.request

def chromeWebdriver():
    options = Options()
    options.add_argument("lang=ko_KR")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extentsions")
    options.add_experimental_option('detach', True)
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecho)' \
                 'Chrome/100.0.4896.75 Safari/537.36'
    options.add_argument(f'user-agent={user_agent}')

    driver = webdriver.Chrome(service=Service(executable_path=ChromeDriverManager().install()), options=options)
    return driver

def collect_images(search_word, extract_img_count):
    url = 'https://www.google.co.kr'

    now = time.localtime()
    today_time = f'{now.tm_year}{now.tm_mon}{now.tm_mday}_{now.tm_hour}{now.tm_min}{now.tm_sec}'
    print(today_time)

    file_path = "c:\\temp\\"

    os.chdir(file_path)
    os.makedirs(file_path + today_time + '_' + search_word)
    os.chdir(file_path + today_time + '_' + search_word)
    file_save_dir = file_path + today_time + '_' + search_word
    print(file_save_dir)

    driver = chromeWebdriver()
    driver.get(url)
    time.sleep(random.uniform(2, 3))
    elem_q = driver.find_element(By.NAME, 'q')
    elem_q.send_keys(search_word)
    elem_q.submit()

    driver.find_element(By.LINK_TEXT, '이미지').click()

    time.sleep(random.uniform(1, 2))

    def page_scrolling(drivers):
        elem = driver.find_element(By.TAG_NAME, 'body')
        page_height = driver.execute_script('return document.body.scrollHeight')

        more_view_cnt = 0
        scroll_cnt = 1
        more_view_scroll_cnt = -1

        equal_cnt = 1
        while True:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(random.uniform(0.3, 0.5))
            new_height = driver.execute_script('return document.body.scrollHeight')
            if page_height != new_height:
                page_height = new_height
                equal_cnt = 1
            print(f'scroll_cnt: {scroll_cnt}, new_height: {new_height}, equal_cnt: {equal_cnt}')

            try:
                scroll_cnt += 1
                equal_cnt += 1
                driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input').click()
                print('결과 더보기 버튼 클릭 처리')
                more_view_scroll_cnt = scroll_cnt
                more_view_cnt += 1
            except:
                if equal_cnt == 20:
                    break
                continue
        ## Endo of scrolling

    page_scrolling(driver)

    file_no = 1
    count = 1
    img_src = []

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    imgs = driver.find_elements(By.CSS_SELECTOR, '#islrg > div.islrc > div a.wXeWr.islib.nfEiy')
    print(len(imgs))

    for img in imgs:
        img_src1 = img.click()
        try:
            img_src2 = driver.find_element(By.CSS_SELECTOR, '#Sva75c > div > div > div.pxAole > div.tvh9oe.BIB1wf > c-wiz > div > div.OUZ5W > div.zjoqD > div.qdnLaf.isv-id > div > a')

        except Exception:
            continue
        time.sleep(random.uniform(0.2, 0.5))
        img_src3 = img_src2.find_element(By.TAG_NAME, 'img').get_attribute('src')
        if img_src3[:4] != 'http':
            continue
        print(count, img_src3, '\n')

        img_src.append(img_src3)
        if count == extract_img_count + 10:
            break
        count += 1

    print()

    for i in range(len(img_src)):
        extention =img_src[i].split('.')[-1]
        ext = ''
        if extention in ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'gif', 'GIF'):
            ext = '.' + extention
        else:
            ext = '.jpg'
        try:
            urllib.request.urlretrieve(img_src[i], str(file_no).zfill(3) + ext)
            print(img_src[i])
        except Exception:
            continue
        print(f'{file_no}번째 이미지 저장-----')
        file_no += 1

        if file_no - 1 == extract_img_count:
            break
    driver.close()

if __name__ == '__main__':
    collect_images('cellphone', 3000)

# https://goodthings4me.tistory.com/783