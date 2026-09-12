from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US, en;q=0.5'
    }
    
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the product title
        title_element = soup.find('span', id='productTitle')
        title = title_element.get_text(strip=True) if title_element else 'Title not found'
        
        # Extract the product image URL
        image_element = soup.find('div', id='imgTagWrapperId')
        image_url = image_element.find('img')['src'] if image_element else 'Image link not found'
        
        # Extract the product description
        description_element = soup.find('div', id='productDescription')
        description = description_element.get_text(strip=True) if description_element else 'Description not found'
        
        # Extract the product price
        price_element = soup.find('span', class_='a-price-whole')
        price = price_element.get_text(strip=True) if price_element else 'Price not found'
        
        return jsonify({
            "title": title,
            "image_url": image_url,
            "description": description,
            "price": price,
            "url":url
        })
    
    else:
        return jsonify({"error": f"Failed to retrieve the page. Status code: {response.status_code}"}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)
