import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import requests

# Yemek tanıma modeli yükleniyor (örneğin mobilenet)
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Kalori veritabanı
calorie_data = {
    "pizza": 266,
    "burger": 295,
    "cheeseburger": 300,
    "sushi": 45,
    "pasta": 131,
    "salad": 33,
    "steak": 271,
    "fried chicken": 320,
    "apple pie": 237,
    "ice_cream": 207,
    "chocolate cake": 352,
    "french fries": 312,
    "omelette": 154,
    "sandwich": 250,
    "donut": 195,
    "taco": 226,
    "spaghetti": 158,
    "lasagna": 135,
    "cheeseburger": 303,
    "hotdog": 151,
    "pancakes": 227,
    "waffles": 291,
    "cereal": 379,
    "bacon": 541,
    "muffin": 364,
    "bagel": 289,
    "brownie": 466,
    "shrimp": 99,
    "lobster": 89,
    "chicken soup": 75,
    "beef stew": 150,
    "mashed potatoes": 88,
    "tuna salad": 194,
    "caesar salad": 158,
    "fish and chips": 343,
    "falafel": 333,
    "hummus": 166,
    "guacamole": 160,
    "quiche": 320,
    "ratatouille": 40,
    "risotto": 202,
    "pudding": 146,
    "biryani": 292,
    "churros": 116,
    "dim sum": 300,
    "gazpacho": 30,
    "paella": 158,
    "ramen": 436,
    "samosa": 262,
    "spring rolls": 64,
    "tiramisu": 240,
    "tofu": 76,
    "udon": 127,
    "yakitori": 203,
}

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def classify_image(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)
    return decoded_preds[0][0][1]  # En olası sınıf ismi

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        # Yemek sınıflandırma
        food_name = classify_image(file_path)
        food_label.config(text="Yemek: " + food_name)

        # Kalori bilgisi
        calories = calorie_data.get(food_name.lower(), "Bilinmiyor")
        calorie_label.config(text="Kalori: " + str(calories))

# GUI kurulum
root = tk.Tk()
root.title("Yemek Tanıma ve Kalori Hesaplama")

upload_button = tk.Button(root, text="Resim Yükle", command=upload_image)
upload_button.pack()

panel = tk.Label(root)
panel.pack()

food_label = tk.Label(root, text="Yemek: ")
food_label.pack()

calorie_label = tk.Label(root, text="Kalori: ")
calorie_label.pack()

root.mainloop()

