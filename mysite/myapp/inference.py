import sys
import numpy as np
import cv2
from keras.models import model_from_json
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class_labels = [
    'basophil',
    'eosinophil',
    'erythroblast',
    'immature granulocytes',
    'lymphocyte',
    'monocyte',
    'neutrophil',
    'platelet'
]

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (28, 28))  
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Ensure the model is built
    _ = model.predict(img_array)
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None. This usually means the last conv layer is not connected to the output, or the model is not built correctly for gradient computation.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    cv2.imwrite(cam_path, superimposed_img)

def load_vgg16_model():
    model_json_path = os.path.join(CURRENT_DIR, 'vgg16.json')
    model_weights_path = os.path.join(CURRENT_DIR, 'vgg16.weights.h5')
    
    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"Model JSON file not found at: {model_json_path}")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights file not found at: {model_weights_path}")
    
    with open(model_json_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    return model

def compute_saliency_map(model, img_array, class_index):
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:, class_index]
    grads = tape.gradient(loss, img_tensor)

    saliency = np.max(np.abs(grads[0]), axis=-1)

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def show_saliency_on_image(img_path, saliency, save_path="saliency_map.png"):
    # Load the original image (high-res)
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape

    
    saliency_resized = cv2.resize(saliency, (w, h))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(orig_img)
    plt.imshow(saliency_resized, cmap='hot', alpha=0.5)
    plt.axis('off')
    plt.title('Saliency Map Overlay')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    # Set the image path here
    img_path = '/Users/aliserromankul/Desktop/testserv/informatrix-project2/mysite/myapp/static/extracted_images/img_012_label_6.jpg'
    model = load_vgg16_model()
    img = preprocess_image(img_path)
    preds = model.predict(img)
    pred_class = np.argmax(preds, axis=1)[0]
    print(f"Predicted class: {class_labels[pred_class]} (index: {pred_class})")
    print(f"Probabilities: {preds[0]}")

    saliency = compute_saliency_map(model, img, pred_class)
    show_saliency_on_image(img_path, saliency)
    