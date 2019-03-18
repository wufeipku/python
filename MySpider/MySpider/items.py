# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html


from scrapy.item import Item, Field

class MyspiderItem(Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    parent_title = Field()
    parent_url = Field()
    second_title = Field()
    second_url = Field()
    path = Field()
    link_title = Field()
    link_url = Field()
    head = Field()
    content = Field()
    pass