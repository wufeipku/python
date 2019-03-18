# -*-coding: utf-8 -*-
__author__ = 'George'
import os
from scrapy.spider import Spider
from scrapy.http import Request
from scrapy.selector import Selector
from MySpider.items import MyspiderItem

base = "d:/dataset/"  # 存放文件分类的目录


class SinaSpider(Spider):
    name = "sina"
    allowed_domains = ["sina.com.cn"]
    start_urls = [
        "http://news.sina.com.cn/guide/"
    ]  # 起始urls列表

    def parse(self, response):
        items = []
        sel = Selector(response)
        big_urls = sel.xpath('//div[@id=\"tab01\"]/div/h3/a/@href').extract()  # 大类的url
        big_titles = sel.xpath("//div[@id=\"tab01\"]/div/h3/a/text()").extract()
        second_urls = sel.xpath('//div[@id=\"tab01\"]/div/ul/li/a/@href').extract()  # 小类的url
        second_titles = sel.xpath('//div[@id=\"tab01\"]/div/ul/li/a/text()').extract()

        for i in range(1, len(big_titles) - 1):  # 这里不想要第一大类,big_title减去1是因为最后一个大类，没有跳转按钮，也去除
            file_name = base + big_titles[i]
            # 创建目录
            if (not os.path.exists(file_name)):
                os.makedirs(file_name)
            for j in range(19, len(second_urls)):
                item = MyspiderItem()
                item['parent_title'] = big_titles[i]
                item['parent_url'] = big_urls[i]
                if_belong = second_urls[j].startswith(item['parent_url'])
                if (if_belong):
                    second_file_name = file_name + '/' + second_titles[j]
                    if (not os.path.exists(second_file_name)):
                        os.makedirs(second_file_name)
                    item['second_url'] = second_urls[j]
                    item['second_title'] = second_titles[j]
                    item['path'] = second_file_name
                    items.append(item)
        for item in items:
            yield Request(url=item['second_url'], meta={'item_1': item}, callback=self.second_parse)

    # 对于返回的小类的url，再进行递归请求
    def second_parse(self, response):
        sel = Selector(response)
        item_1 = response.meta['item_1']
        items = []
        bigUrls = sel.xpath('//a/@href').extract()

        for i in range(0, len(bigUrls)):
            if_belong = bigUrls[i].endswith('.shtml') and bigUrls[i].startswith(item_1['parent_url'])
            if (if_belong):
                item = MyspiderItem()
                item['parent_title'] = item_1['parent_title']
                item['parent_url'] = item_1['parent_url']
                item['second_url'] = item_1['second_url']
                item['second_title'] = item_1['second_title']
                item['path'] = item_1['path']
                item['link_url'] = bigUrls[i]
                items.append(item)
        for item in items:
            yield Request(url=item['link_url'], meta={'item_2': item}, callback=self.detail_parse)

    def detail_parse(self, response):
        sel = Selector(response)
        item = response.meta['item_2']
        content = ""
        head = sel.xpath('//h1[@id=\"artibodyTitle\"]/text()').extract()
        content_list = sel.xpath('//div[@id=\"artibody\"]/p/text()').extract()
        for content_one in content_list:
            content += content_one
        item['head'] = head
        item['content'] = content
        yield item
