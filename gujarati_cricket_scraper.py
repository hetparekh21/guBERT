import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime

class GujaratiCricketScraper:
    def __init__(self, base_url="https://gujarati.abplive.com/sports", start_page=2, end_page=447):
        self.base_url = base_url
        self.start_page = start_page
        self.end_page = end_page
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9,gu;q=0.8'
        }
        self.data = []
        
        # Create directory for saving data
        self.output_dir = "gujarati_cricket_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def get_article_links(self, page_url):
        """Extract all article links from the nested structure: story-wrapper > pagination > anchors"""
        try:
            response = requests.get(page_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the story-wrapper section first
            story_wrapper = soup.find('section', class_='story-wrapper')
            if not story_wrapper:
                print(f"Could not find story-wrapper section on {page_url}")
                return []
            
            # Find the pagination section inside the story-wrapper
            pagination_section = story_wrapper.find('section', class_='pagination')
            if not pagination_section:
                print(f"Could not find pagination section on {page_url}")
                return []
                
            # Find all article links within the pagination section
            article_links = []
            article_anchors = pagination_section.find_all('a')
            
            for anchor in article_anchors:
                if 'href' in anchor.attrs:
                    # Skip page navigation links - they usually contain 'page-'
                    if 'page-' not in anchor['href']:
                        article_links.append(anchor['href'])
            
            print(f"Found {len(article_links)} articles on {page_url}")
            return article_links
            
        except Exception as e:
            print(f"Error fetching article links from {page_url}: {e}")
            return []
    
    def extract_article_content(self, article_url, page_number):
        """Extract content from an individual article"""
        try:
            # Check if the URL is absolute or relative
            if not article_url.startswith('http'):
                article_url = 'https://gujarati.abplive.com' + article_url
                
            response = requests.get(article_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = ""
            title_tag = soup.find('h1', class_='article-heading')
            if title_tag:
                title = title_tag.text.strip()
            
            # Extract date
            date = ""
            date_tag = soup.find('span', class_='article-publish-date')
            if date_tag:
                date = date_tag.text.strip()
            
            # Extract main content from div with class "abp-story-detail"
            content = ""
            story_detail = soup.find('div', class_='abp-story-detail')
            if story_detail:
                paragraphs = story_detail.find_all('p')
                content = '\n\n'.join([p.text.strip() for p in paragraphs if p.text.strip()])
            else:
                # Fallback to article-content if abp-story-detail is not found
                article_body = soup.find('div', class_='article-content')
                if article_body:
                    paragraphs = article_body.find_all('p')
                    content = '\n\n'.join([p.text.strip() for p in paragraphs if p.text.strip()])
            
            # Get current timestamp
            scrape_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Return extracted data
            return {
                'title': title,
                'date': date, 
                'content': content,
                'url': article_url,
                'page_number': page_number,
                'scrape_timestamp': scrape_timestamp
            }
            
        except Exception as e:
            print(f"Error extracting content from {article_url}: {e}")
            return None
    
    def scrape_data(self):
        """Main function to scrape data"""
        total_articles = 0
        
        for page_num in range(self.start_page, self.end_page + 1):
            page_url = f"{self.base_url}/page-{page_num}"
            print(f"Processing page {page_num} of {self.end_page}")
            
            # Get all article links from the current page
            article_links = self.get_article_links(page_url)
            
            # Process each article
            for link in article_links:
                article_data = self.extract_article_content(link, page_num)
                if article_data:
                    self.data.append(article_data)
                    total_articles += 1
                    print(f"Scraped article {total_articles}: {article_data['title'][:50]}...")
                
                # Be polite to the server with a random delay
                time.sleep(2)
            
            # Save data periodically to avoid losing progress
            if page_num % 5 == 0 or page_num == self.end_page:
                self.save_data()
                
            # Longer delay between pages
            # time.sleep(random.uniform(3, 5))
        
        print(f"Completed scraping {total_articles} articles across {self.end_page - self.start_page + 1} pages.")
    
    def save_data(self):
        """Save the scraped data to CSV and JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        df = pd.DataFrame(self.data)
        csv_filename = f"{self.output_dir}/gujarati_cricket_data_{timestamp}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        # Save as JSON
        json_filename = f"{self.output_dir}/gujarati_cricket_data_{timestamp}_{self.start_page}_to_{self.end_page}.json"
        df.to_json(json_filename, orient='records', force_ascii=False, indent=4)
        
        print(f"Data saved to {csv_filename} and {json_filename}")
        print(f"Collected {len(self.data)} articles so far")

def main():
    # Set the range of pages you want to scrape
    # For testing, you might want to start with a smaller range
    start_page = 21
    end_page = 200  # Change this to a smaller number for testing, like 5
    
    scraper = GujaratiCricketScraper(start_page=start_page, end_page=end_page)
    scraper.scrape_data()
    
    print("Scraping completed!")

if __name__ == "__main__":
    main()
