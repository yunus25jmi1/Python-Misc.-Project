import requests
from bs4 import BeautifulSoup

def get_html(query):
  url = f"https://www.google.com/search?q={query}"
  response = requests.get(url)
  return response.content

def parse_html(html):
  soup = BeautifulSoup(html, "html.parser")

  # Find the product price
  price = soup.find("span", class_="a-offscreen").text

  # Find other product details
  title = soup.find("span", id="productTitle").text
  description = soup.find("div", id="productDescription").text
  image_url = soup.find("img", id="imgBlkFront").get("src")

  return {
    "price": price,
    "title": title,
    "description": description,
    "image_url": image_url,
  }

def scrape_product_details(product):
  html = get_html(product)
  product_details = parse_html(html)
  return product_details

def main():
  product = "iPhone 13"
  product_details = scrape_product_details(product)

  print(f"Product: {product_details['title']}")
  print(f"Price: {product_details['price']}")
  print(f"Description: {product_details['description']}")
  print(f"Image URL: {product_details['image_url']}")

if __name__ == "__main__":
  main()
