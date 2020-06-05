#!/usr/bin/env python
# coding: utf-8

# In[1]:


#-*- coding:utf-8 -*-

import requests as rq
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urljoin
import webtoon_config as WC
import json


# In[2]:


def save(data, file_name):
    file = open(file_name, 'a')
    file.write(data + '\n')


# In[3]:


def get_daylywebtoons():
    '''
    요일 웹툰을 수집
    '''
    webtoon_main_url = WC.TOP_URL
    res = rq.get(webtoon_main_url)
    main_soup = BeautifulSoup(res.content, 'lxml')
 
    webtoon_links = [{"title": a_tag.get('title'), "link": urljoin(WC.NAVER_URL, a_tag.get('href'))}
                      for a_tag in main_soup.select('.daily_all a.title')]
 
    return webtoon_links


# In[4]:


def make_link(webtoon_url, page_count):
    return webtoon_url + '&page=' + str(page_count)


# In[5]:


def get_all_webtoon(webtoon, is_save):
    '''
    해당 웹 툰의  1화 ~ 마지막까지 수집
    '''
    page_count = 1
    is_unlast = True
 
    target_webtoons = list()
    webtoon_url = webtoon['link']
    webtoon_title = webtoon['title']
 
    while is_unlast:
        link = make_link(webtoon_url, page_count)
 
        target_webtoon_res = rq.get(link)
        webtoon_soup = BeautifulSoup(target_webtoon_res.content, 'lxml')
        a_tags = webtoon_soup.select('.viewList td.title a')
 
        for a_tag in a_tags:
            t = a_tag.text.replace('\n', '').replace('\r', '').replace('\t', '')
            h = urljoin(WC.NAVER_URL, a_tag.get('href'))
 
            if h not in target_webtoons:
                target_webtoons.append(h)
            else:
                is_unlast = False
 
        page_count += 1
 
    if is_save:
        for webtoon in target_webtoons:
            save(webtoon_title + ':' + webtoon, 'all_webtoons.txt')
 
    return target_webtoons


# In[6]:


def data_parse(soup, url, is_save, page_count):
    rank = soup.select('#topPointTotalNumber')[0].text
    title = soup.title.text.split(':')[0]
 
    titleId = str(730425)
    no = str(5)
 
    comment_url = WC.NAVER_URL + '/comment/comment.nhn?titleId=' + titleId + '&no=' + no
    objectId = titleId + '_' + no

    
 
    u = 'http://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json?ticket=comic&templateId=webtoon&pool=cbox3&_callback=jQuery1113012327623800394427_1489937311100&lang=ko&country=KR&objectId=' +objectId+ '&categoryId=&pageSize=15&indexSize=10&groupId=&listType=OBJECT&sort=NEW&_=1489937311112'
 
    while True:
        comment_url = make_link(u, page_count)
        header = {
            "Host": "apis.naver.com",
            "Referer": "http://comic.naver.com/comment/comment.nhn?titleId=" + titleId + "&no=" + no,
            "Content-Type": "application/javascript"
        }
 
        res = rq.get(comment_url, headers = header)
        soup = BeautifulSoup(res.content, 'lxml')
        try:
            content_text = soup.select('p')[0].text
            one = content_text.find('(') + 1
            two = content_text.find(');')
            content = json.loads(content_text[one:two])
 
            comments = content['result']['commentList']
 
            print(page_count)
            for comment in comments:
                #print(comment['contents'])
                if is_save:
                    save(comment['contents'], '판타지여동생.txt')
 
            if not len(comments):
                break
            else:
                page_count -= 1
        except:
            pass


# In[206]:


if __name__ == "__main__":
    webtoons = get_daylywebtoons()
    for webtoon in webtoons:
        target_webtoons = get_all_webtoon(webtoon, False)
        for webtoon_page in target_webtoons:
            res = rq.get(webtoon_page)
            webtoon_page_soup = BeautifulSoup(res.content, 'lxml')
            data_parse(webtoon_page_soup, webtoon_page, True, 16)


# In[ ]:


NAVER_URL = 'http://comic.naver.com'
TOP_URL = 'http://comic.naver.com/webtoon/finish.nhn'

