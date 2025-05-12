import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    name = 'narutospider'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
    }

    def parse(self, response):
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
            yield scrapy.Request("https://naruto.fandom.com" + href, callback=self.parse_jutsu)

        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)
    
    def parse_jutsu(self, response):
        jutsu_name = response.css("span.mw-page-title-main::text").extract()[0].strip()
        div_selector = response.css("div.mw-parser-output")[0]
        div_html = div_selector.extract()
        soup = BeautifulSoup(div_html, "html.parser").find('div')

        jutsu_type = ""
        if soup.find('aside'):
            aside = soup.find('aside')
            for cell in aside.find_all('div', {'class': 'pi-data'}):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == "Classification":
                        jutsu_type = cell.find('div').text.strip()

            soup.find('aside').decompose()

        jutsu_description = soup.text.strip().split('Trivia')[0].strip()

        yield {
            "jutsu_name": jutsu_name,
            "jutsu_type": jutsu_type,
            "jutsu_description": jutsu_description
        }
