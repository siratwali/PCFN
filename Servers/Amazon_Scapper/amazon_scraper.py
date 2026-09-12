import requests
from bs4 import BeautifulSoup
import time

def fetch_amazon_products():
    url = 'https://www.amazon.com/s?k=Milk&ref=cs_503_search'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    proxies = {
        "http": "http://192.168.1.1:8080",  # Replace with your actual proxy IP and port
        "https": "https://192.168.1.1:8080",
    }

    # Introduce a delay to avoid getting blocked
    time.sleep(5)  # Sleep for 5 seconds before making the request

    try:
        response = requests.get(url, headers=headers, proxies=proxies)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            products = []

            for item in soup.select('.s-result-item'):
                name = item.select_one('h2 .a-text-normal')
                price = item.select_one('.a-price-whole')
                image = item.select_one('.s-image')

                if name and price and image:
                    products.append({
                        'name': name.get_text().strip(),
                        'price': price.get_text().strip(),
                        'image': image['src']
                    })

                if len(products) >= 10:
                    break

            return products
        else:
            raise Exception(f'Failed to load products. Status Code: {response.status_code}')

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    try:
        products = fetch_amazon_products()
        for product in products:
            print(product)
    except Exception as e:
        print(f"Error: {str(e)}")
