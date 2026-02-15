# import torch
# from torchvision.models import densenet121, DenseNet121_Weights
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import random
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from skimage.transform import resize
# from matplotlib import pyplot as plt
# import os

# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

# set_seed(42)

# weights = DenseNet121_Weights.IMAGENET1K_V1
# model = densenet121(weights=weights)
# model.classifier = torch.nn.Linear(model.classifier.in_features, 14)  
# model.eval()

# chexnet_classes = [
#     "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
#     "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
#     "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
# ]

# thresholds = {
#     "Atelectasis": 0.7, "Cardiomegaly": 0.7, "Effusion": 0.75, "Infiltration": 0.7,
#     "Mass": 0.7, "Nodule": 0.7, "Pneumonia": 0.65, "Pneumothorax": 0.7,
#     "Consolidation": 0.65, "Edema": 0.7, "Emphysema": 0.75, "Fibrosis": 0.75,
#     "Pleural Thickening": 0.7, "Hernia": 0.75
# }

# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# data_augmentation = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def validate_image_file(image_path):
#     valid_extensions = {".png", ".jpg", ".jpeg"}
#     _, ext = os.path.splitext(image_path)
#     if ext.lower() not in valid_extensions:
#         raise ValueError(f"Unsupported file format: {ext}. Please use one of {valid_extensions}.")

# def validate_image_size(image):
#     min_size = (500, 500)
#     max_size = (3000, 3000)
#     if not (min_size[0] <= image.width <= max_size[0] and min_size[1] <= image.height <= max_size[1]):
#         raise ValueError(f"Image size {image.size} is not within the valid range {min_size} - {max_size}.")

# def analyze_image(image_path, use_augmentation=False):
#     validate_image_file(image_path)

#     image = Image.open(image_path).convert("RGB")

#     validate_image_size(image)

#     if use_augmentation:
#         input_tensor = data_augmentation(image).unsqueeze(0)
#     else:
#         input_tensor = preprocess(image).unsqueeze(0)

    
#     output = model(input_tensor)
#     predicted_probs = torch.sigmoid(output).detach().numpy()[0]

#     detected_diseases = []
#     print("Detected diseases with probabilities:")
#     for i, prob in enumerate(predicted_probs):
#         if prob > thresholds[chexnet_classes[i]]:
#             detected_diseases.append((chexnet_classes[i], prob * 100))
#             print(f"- {chexnet_classes[i]}: {prob * 100:.2f}%")
#     return input_tensor, image, detected_diseases

# def visualize_gradcam(input_tensor, image, detected_diseases):
#     if not detected_diseases:
#         print("No diseases detected. Skipping Grad-CAM visualization.")
#         return

#     target_layer = model.features[-2]  
#     cam = GradCAM(model=model, target_layers=[target_layer])

#     grayscale_cam = cam(input_tensor=input_tensor, targets=None)
#     grayscale_cam = grayscale_cam[0, :]  

#     grayscale_cam_resized = resize(grayscale_cam, (image.size[1], image.size[0]))

#     img_array = np.array(image) / 255.0

#     visualization = show_cam_on_image(img_array, grayscale_cam_resized)

#     plt.imshow(visualization)
#     plt.axis('off')
#     plt.title("Grad-CAM Visualization")
#     plt.show()

# if __name__ == "__main__":
#     image_path = "chest-xray9.png"  #

#     input_tensor, image, detected_diseases = analyze_image(image_path, use_augmentation=True)

#     visualize_gradcam(input_tensor, image, detected_diseases)