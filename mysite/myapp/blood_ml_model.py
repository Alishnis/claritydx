import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Функция для загрузки модели и весов
def load_model_and_weights(model_json_path, model_weights_path):
    # Загрузка структуры модели из файла JSON
    with open(model_json_path, 'r') as json_file:
        model_structure = json_file.read()
    
    model = model_from_json(model_structure)
    model.load_weights(model_weights_path)  # Загрузка весов
    return model

# Функция для предобработки изображения
def preprocess_image(image_path, target_size=(28, 28)):
    img = image.load_img(image_path, target_size=target_size)  # Изменение размера изображения
    img_array = img_to_array(img)  # Преобразование изображения в массив
    img_array = np.expand_dims(img_array, axis=0)  # Добавление размерности для батча
    img_array = img_array / 255.0  # Нормализация изображения
    return img_array

# Функция для анализа изображения
def predict_image(model, image_path, class_names):
    # Предобработка изображения
    img_array = preprocess_image(image_path)
    
    # Получение предсказания от модели
    prediction = model.predict(img_array)
    
    # Получение класса с наибольшей вероятностью
    predicted_class = np.argmax(prediction, axis=-1)
    
    # Получение названия заболевания по предсказанному классу
    predicted_disease = class_names[predicted_class[0]]
    
    return predicted_disease, predicted_class, prediction

# Загрузка модели и весов
model = load_model_and_weights('model_from_scratch_blood.json', 'model_from_scratch_blood.weights.h5')

# Список заболеваний (предполагается, что у вас 8 классов, измените список по своему усмотрению)
class_names = ['basophil',
                'eosinophil',
                'erythroblast',
                'immature granulocytes',
                'lymphocyte',
                'monocyte',
                'neutrophil',
                'platelet']

# Путь к изображению для анализа
image_path = '/Users/aliserromankul/Desktop/testserv/informatrix-project2/mysite/myapp/static/extracted_images/img_012_label_6.jpg'  # Укажите путь к вашему изображению

# Анализ изображения
predicted_disease, predicted_class, prediction = predict_image(model, image_path, class_names)

# Отображение изображения
img = image.load_img(image_path)
plt.imshow(img)
plt.show()

# Вывод результатов
print(f"Predicted Disease: {predicted_disease}")  # Вывод названия заболевания
print(f"Predicted Class: {predicted_class[0]}")  # Вывод числового класса
print(f"Prediction Probabilities: {prediction}")  # Вывод вероятностей для каждого класса