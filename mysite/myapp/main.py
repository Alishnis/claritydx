from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('/Users/aliserromankul/Desktop/lessons/hackathon/mysite/myapp/almas.py')

def preprocess_image(image_path, target_size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)

    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

image_path = "/Users/aliserromankul/Desktop/lessons/hackathon/mysite/data/test/adenocarcinoma/test.png"

input_data = preprocess_image(image_path, target_size=(224, 224)) 

predictions = model.predict(input_data)
print("Предсказания:", predictions)