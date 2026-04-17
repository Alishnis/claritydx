import os
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from .inference import preprocess_image, load_vgg16_model, compute_saliency_map, show_saliency_on_image, class_labels
from .models import BloodCellAnalysis
import numpy as np
import pandas as pd
import pickle
import ast
import logging
from django.core.files import File
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

def upload_image(request):
    if request.method == 'POST':
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'Изображение не загружено'}, status=400)
        
        image = request.FILES['image']
        
        # Create a temporary file path
        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_upload.jpg')
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        # Save the uploaded file temporarily
        with open(temp_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        
        try:
            # Load and preprocess the image
            img = preprocess_image(temp_path)
            
            # Load model and make prediction
            model = load_vgg16_model()
            preds = model.predict(img)
            pred_class = np.argmax(preds, axis=1)[0]
            
            # Compute saliency map
            saliency = compute_saliency_map(model, img, pred_class)
            
            # Save saliency map
            saliency_path = os.path.join(settings.MEDIA_ROOT, 'saliency_maps', f'saliency_map_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            os.makedirs(os.path.dirname(saliency_path), exist_ok=True)
            show_saliency_on_image(temp_path, saliency, save_path=saliency_path)
            
            # Save original image
            image_path = os.path.join(settings.MEDIA_ROOT, 'blood_cell_images', f'blood_cell_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            with open(temp_path, 'rb') as src, open(image_path, 'wb') as dst:
                dst.write(src.read())
            
            # Create database entry
            analysis = BloodCellAnalysis(
                user=request.user if request.user.is_authenticated else None,
                prediction=class_labels[pred_class],
                confidence=float(preds[0][pred_class])
            )
            
            # Save the image files
            with open(image_path, 'rb') as f:
                analysis.image.save(os.path.basename(image_path), File(f), save=False)
            with open(saliency_path, 'rb') as f:
                analysis.saliency_map.save(os.path.basename(saliency_path), File(f), save=False)
            
            analysis.save()
            
            # Prepare response
            result = {
                'prediction': class_labels[pred_class],
                'confidence': float(preds[0][pred_class]),
                'all_probabilities': {label: float(prob) for label, prob in zip(class_labels, preds[0])},
                'saliency_map_url': f'/media/saliency_maps/{os.path.basename(saliency_path)}'
            }
            
            # Clean up temporary files
            os.remove(temp_path)
            
            return render(request, 'blood_cell_result.html', {
                'prediction': result['prediction'],
                'confidence': f"{result['confidence']*100:.2f}%",
                'all_probabilities': result['all_probabilities'],
                'saliency_map_url': result['saliency_map_url'],
                'show_save_button': True
            })
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Error in upload_image: {str(e)}", exc_info=True)
            return render(request, 'upload_cellblood.html', {'error': str(e)})
    
    return render(request, 'upload_cellblood.html')



# Load the recommendation system data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECOMMENDATION_DIR = os.path.join(BASE_DIR, 'myapp', 'recommendation system')

# Load datasets
sym_des = pd.read_csv(os.path.join(RECOMMENDATION_DIR, "datasets/symtoms_df.csv"))
precautions = pd.read_csv(os.path.join(RECOMMENDATION_DIR, "datasets/precautions_df.csv"))
workout = pd.read_csv(os.path.join(RECOMMENDATION_DIR, "datasets/workout_df.csv"))
description = pd.read_csv(os.path.join(RECOMMENDATION_DIR, "datasets/description.csv"))
medications = pd.read_csv(os.path.join(RECOMMENDATION_DIR, 'datasets/medications.csv'))
diets = pd.read_csv(os.path.join(RECOMMENDATION_DIR, "datasets/diets.csv"))

# Load model
svc = pickle.load(open(os.path.join(RECOMMENDATION_DIR, 'models/svc.pkl'), 'rb'))

# Helper functions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def get_predicted_value(patient_symptoms):
    
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    
 
    input_df = pd.DataFrame([input_vector], columns=list(symptoms_dict.keys()))
    
    prediction = svc.predict(input_df)[0]
    return diseases_list[prediction]

def recommendation_view(request):
    if request.method == 'POST':
        selected_symptoms = request.POST.getlist('symptoms')
        
        if not selected_symptoms:
            return render(request, 'recommendation.html', {
                'message': "Пожалуйста, выберите хотя бы один симптом",
                'valid_symptoms': list(symptoms_dict.keys())
            })
        
        try:
            logger.info(f"Processing symptoms: {selected_symptoms}")
            
            for symptom in selected_symptoms:
                if symptom not in symptoms_dict:
                    raise ValueError(f"Недействительный симптом: {symptom}")
            
            predicted_disease = get_predicted_value(selected_symptoms)
            logger.info(f"Predicted disease: {predicted_disease}")
            
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            logger.info("Helper function completed successfully")

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)
            
            try:
                medications = ast.literal_eval(medications)
                rec_diet = ast.literal_eval(rec_diet)
            except (SyntaxError, ValueError) as e:
                logger.error(f"Error parsing medications or diet: {str(e)}")
                medications = medications if isinstance(medications, list) else [medications]
                rec_diet = rec_diet if isinstance(rec_diet, list) else [rec_diet]
            medications = ast.literal_eval(medications[0])
            rec_diet = ast.literal_eval(rec_diet[0])

            return render(request, 'recommendation.html', {
                'predicted_disease': predicted_disease,
                'dis_des': dis_des,
                'my_precautions': my_precautions,
                'my_medications': medications,
                'my_diet': rec_diet,
                'workout': workout,
                'valid_symptoms': list(symptoms_dict.keys())
            })
        except Exception as e:
            logger.error(f"Error in recommendation_view: {str(e)}", exc_info=True)
            return render(request, 'recommendation.html', {
                'message': f"Произошла ошибка: {str(e)}",
                'valid_symptoms': list(symptoms_dict.keys())
            })

    return render(request, 'recommendation.html', {
        'valid_symptoms': list(symptoms_dict.keys())
    })
