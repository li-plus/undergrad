import json
import os

import scrapy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class NewsSpider(object):

    def collect_news_urls(self):
        nids = [113352,
                113322,
                113205,
                113667,
                11139635,
                11147664,
                11121835,
                11118384,
                116727,
                11142780,
                11154139,
                1160570,
                11123639,
                1158394,
                113889,
                113581,
                1184874,
                11109063,
                11109394,
                11111110,
                11109466,
                11109975,
                11111389,
                11164734,
                11164735,
                11135735,
                11148492,
                11157485,
                114322,
                11148341,
                1118220,
                1171880,
                11110873,
                11110779, ]

        news_urls = []

        urls = [
            "http://qc.wa.news.cn/nodeart/list?nid={nid}&pgnum={pgnum}&cnt=1000&tp=1&orderby=1".format(
                nid=nid, pgnum=pgnum)
            for nid in nids for pgnum in range(1, 6)]

        chromeOptions = webdriver.ChromeOptions()
        chromeOptions.add_argument("--headless")
        chromeOptions.add_argument("--disable-gpu")
        self.driver = webdriver.Chrome(
            executable_path="C:/Users/80685/AppData/Local/Google/Chrome/Application/chromedriver.exe",
            options=chromeOptions)

        print(len(urls))
        print(len(set(urls)))

        for json_url in urls:
            try:
                self.driver.get(json_url)
                self.driver.implicitly_wait(3)
                print(self.driver.current_url)

                sel = scrapy.Selector(text=self.driver.page_source)

                data = sel.css("body::text").extract_first()[1:-1]
                '''print(res)
                if not len(res):
                    print('re fail')
                    continue
                data = res[0]'''

                datadict = json.loads(data, encoding='utf-8')

                if 'data' in datadict.keys() and 'list' in datadict[
                    'data'].keys():
                    for single_news_info in datadict['data']['list']:
                        if single_news_info['LinkUrl'] not in news_urls:
                            news_urls.append(single_news_info['LinkUrl'])
            except Exception as e:
                print(self.driver.current_url, "decode failed")

        print(len(news_urls))
        print(len(set(news_urls)))

        with open('news_urls.txt', 'w') as f:
            f.write(str(news_urls))

    def parse_news(self):
        with open('news_urls.txt', 'r') as f:
            news_urls = eval(f.read())
        print(len(news_urls))
        print(type(news_urls))

        os.chdir('news_json')
        chromeOptions = webdriver.ChromeOptions()
        chromeOptions.add_argument("--headless")
        chromeOptions.add_argument("--disable-gpu")
        self.driver = webdriver.Chrome(options=chromeOptions)

        for i in range(20809, len(news_urls)):
            if os.path.isfile('{i}.json'.format(i=i)):
                print('{i}.json'.format(i=i), 'file already created')
                continue
            try:
                self.driver.get(news_urls[i])
                self.driver.implicitly_wait(1)
                WebDriverWait(self.driver, 2, 0.2).until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, '#p-detail')))
                sel = scrapy.Selector(text=self.driver.page_source)
                title = sel.css(
                    'div.h-title::text').extract_first()
                '''
                title:
                    body > div.container.cont > div:nth-child(1) > div.tit::text
                content:
                    div.box.txtcont p::text
                time:
                    body > div.container.cont > div:nth-child(2) > div > div.source > span.time::text
                '''
                para_list = sel.css("#p-detail p::text").extract()
                source = sel.css("#source::text").extract_first() or ' '

                pubtime = sel.css(
                    "span.h-time::text").extract_first()
                if len(title) == 0 or len(para_list) == 0 or len(pubtime) == 0:
                    raise Exception("this is not a news page")

                single_news_dict = {'id': i,
                                    'title': title.strip(),
                                    'pubtime': pubtime.strip(),
                                    'content': '\n'.join(para_list).strip(),
                                    'source': source.strip(),
                                    'link': self.driver.current_url}
                with open('{i}.json'.format(i=i), 'w', encoding='utf-8') as f:
                    json.dump(single_news_dict, f, ensure_ascii=False)
                print(i, "news downloaded")
            except Exception as e:
                print(i, "news downloaded fail", self.driver.current_url)
                print("Exception:", e)

    def teardown(self):
        self.driver.quit()


if __name__ == "__main__":
    cs = NewsSpider()
    cs.parse_news()
    cs.teardown()
