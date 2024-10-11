import os
import sys
import math
import random
import json
import re
import datetime

# Product class with getter and setter
class Product:
    def __init__(self, name, price):
        self.name = name
        self._price = price
    
    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, value):
        if value < 0:
            raise ValueError("Price cannot be negative")
        self._price = value

# DiscountedProduct class demonstrates overriding
class DiscountedProduct(Product):
    def __init__(self, name, price, discount):
        super().__init__(name, price)
        self.discount = discount
    
    @property
    def price(self):
        return math.floor(self._price * (1 - self.discount / 100))

# Exception handling example for product not found
class ProductNotFoundError(Exception):
    pass

# Log purchase using a decorator
def log_purchase(func):
    def wrapper(*args, **kwargs):
        print(f"Logging purchase for {args[0]}")
        return func(*args, **kwargs)
    return wrapper

# Regular expression to validate email
def is_valid_email(email):
    pattern = r"[^@]+@[^@]+\.[^@]+"
    return re.match(pattern, email)

# Simulating an ecommerce system
class EcommerceSystem:
    def __init__(self):
        self.products = []
    
    def add_product(self, product):
        self.products.append(product)
    
    def list_products(self):
        return [f"Product: {p.name}, Price: ${p.price}" for p in self.products]
    
    @log_purchase
    def buy_product(self, email, product_name):
        if not is_valid_email(email):
            raise ValueError("Invalid email address")
        
        for product in self.products:
            if product.name == product_name:
                print(f"Product '{product_name}' purchased successfully!")
                return
        raise ProductNotFoundError(f"Product '{product_name}' not found")

# JSON module example for saving products
def save_products_to_file(products):
    product_data = [{"name": p.name, "price": p.price} for p in products]
    with open('products.json', 'w') as json_file:
        json.dump(product_data, json_file)

# OS module to create product images directory
def create_product_image_dir():
    if not os.path.exists("product_images"):
        os.makedirs("product_images")

# DateTime module to get current time
def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Random module to generate order ID
def generate_order_id():
    return random.randint(100000, 999999)

# Main Program
if __name__ == "__main__":
    ecommerce = EcommerceSystem()

    # Adding products
    ecommerce.add_product(Product("Laptop", 1000))
    ecommerce.add_product(Product("Phone", 500))
    
    # List products
    print("Available Products:", ecommerce.list_products())

    try:
        # Buy a product
        ecommerce.buy_product("customer@example.com", "Laptop")
    except Exception as e:
        print(e)

    # Save products to JSON
    save_products_to_file(ecommerce.products)

    # Get current time and order ID
    print(f"Current Time: {get_current_time()}")
    print(f"Generated Order ID: {generate_order_id()}")
