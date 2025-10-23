import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

base_dir = '/Users/aliserromankul/Desktop/lessons/hackathon/mysite/data'
batch_size = 32
img_size = (128, 128)
model_path = "trained_model.h5"

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    os.path.join(base_dir, 'valid'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

if os.path.exists(model_path):
    model = load_model(model_path)
    print("Модель успешно загружена из 'trained_model.h5'")
else:
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(train_generator, 
                        validation_data=valid_generator, 
                        epochs=15, 
                        callbacks=[early_stopping])

    
    model.save(model_path)
    print(f"Модель сохранена в '{model_path}'")


img = load_img("/Users/aliserromankul/Desktop/lessons/hackathon/mysite/data/test/adenocarcinoma/test.png", target_size=(128, 128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  
predictions = model.predict(img_array)
predictions = np.argmax(predictions, axis=1)
print(f"Предсказания: {predictions}")

    

