# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import sys
from scrapy import signals
import json
import codecs

class MyspiderPipeline(object):
    def process_item(self, item, spider):
        link_url = item['link_url']
        file_name = link_url[7:-6].replace('/', '_')
        file_name += ".txt"
        fp = open(item['path'] + '/' + file_name, 'w',encoding='utf8')
        fp.write(item['content'])
        fp.close()
        return item