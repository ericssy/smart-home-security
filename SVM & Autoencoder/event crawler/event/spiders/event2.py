#coding=utf-8
import scrapy
from scrapy.spiders import Spider
from scrapy.crawler import CrawlerProcess
from event import items
import os

class event2(Spider):
    name = "event"
    id = 0

    def start_requests(self):
        print(os.getcwd())
        base_url = "file:///Users/ShenSiyuan/Google%20Drive/Research/Smart%20Home%20Security%20Research/SVM%20&%20Autoencoder/event%20crawler/events_list_motion_sensor.htm"
        yield scrapy.Request(base_url, self.parse)

    def parse(self, response):
        # outputs title and content of an article
        posts = response.xpath("//tbody[@class='events-table']//tr")
        print(len(posts))
        for p in posts:
            item = items.EventItem()
            date = p.xpath(".//span[@class='eventDate']/text()").extract_first()
            date = date.strip().split(" ")
            date = date[0]+" "+date[1]+" "+date[2]
            item["time"] = date
            others = p.xpath(".//a[@class='tooltip-init']/text()").extract()
            item["name"] = others[0].strip()
            item["value"] = others[1].strip()
            yield item

def main(events, context):
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
        'FEED_FORMAT': 'txt',
        #'FEED_URI': 'tmp/id.txt'
    })

    process.crawl(event2)
    process.start()  # the script will block here until the crawling is finished
    # data = open('tmp/id.txt', 'r')
    # s3.put_object(Bucket=BUCKET, Key='result.json', Body=data)

if __name__ == "__main__":
    main('', '')
