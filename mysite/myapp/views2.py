from django.shortcuts import render, redirect
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4
from PIL import Image
import numpy as np
from django.conf import settings
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.transform import resize
from .models import AnalysisSkin, BloodAnalysis
from django.http import JsonResponse



preprocess = transforms.Compose([
    transforms.Resize((380, 380)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_skin_disease_model():
    model = efficientnet_b4(pretrained=True) 
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)  
    model = nn.Sequential(
        model,
        nn.Dropout(0.5)  
    )
    model.eval()
    return model

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, input_tensor, target_layer, predicted_index):
    
    targets = [ClassifierOutputTarget(predicted_index)]
    
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  
    
    return grayscale_cam
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget




def analyze_skin_image(request):
   
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']

        temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)
        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
        with open(temp_image_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        try:
            image = Image.open(temp_image_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0)

            model = load_skin_disease_model()

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            predicted_index = probabilities.argmax().item()
            predicted_class = CLASSES[predicted_index]
            predicted_probability = probabilities[predicted_index].item() * 140

            target_layer = model[0].features[-1]  
            heatmap = generate_gradcam(model[0], input_tensor, target_layer, predicted_index)

            
            heatmap_resized = np.array(Image.fromarray(heatmap).resize(image.size, Image.LANCZOS))
            heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
            heatmap_colored = np.uint8(255 * heatmap_normalized)
            heatmap_overlay = (np.array(image) * 0.5 + np.expand_dims(heatmap_colored, axis=2) * 0.5).astype(np.uint8)

            gradcam_dir = os.path.join(settings.MEDIA_ROOT, 'gradcam')
            os.makedirs(gradcam_dir, exist_ok=True)
            gradcam_path = os.path.join(gradcam_dir, f"gradcam_{os.path.basename(uploaded_file.name)}")
            Image.fromarray(heatmap_overlay).save(gradcam_path)
            analysis = AnalysisSkin.objects.create(
                user=request.user,
                analysis_file=gradcam_path.replace(settings.MEDIA_ROOT, ''),  
                result=f'{predicted_class}: {predicted_probability:.2f}%'
            )

            os.remove(temp_image_path)

            return render(request, 'result_forskin.html', {
                'predicted_class': predicted_class,
                'predicted_probability': f"{predicted_probability:.2f}%",
                'analysis_id': analysis.id,
                'gradcam_url': gradcam_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
            })

        except Exception as e:
            return render(request, 'skin.html', {'error': f"Ошибка анализа: {str(e)}"})

    return render(request, 'skin.html')

CLASSES = [
    "Melanoma", "Nevus", "Basal Cell Carcinoma",
    "Actinic Keratosis", "Benign Keratosis",
    "Dermatofibroma", "Vascular Lesion"
]

from django.contrib import messages

def save_results_skin(request):
   
    if request.method == 'POST':
        analysis_id = request.POST.get('analysis_id')
        try:
            analysis = AnalysisSkin.objects.get(id=analysis_id, user=request.user)
            analysis.is_saved = True 
            analysis.save()
            messages.success(request, 'Результаты успешно сохранены!')
        except AnalysisSkin.DoesNotExist:
            messages.error(request, 'Анализ не найден.')

        return redirect('user_kab')  
    return JsonResponse({'error': 'Недопустимый метод запроса'}, status=405)


from django.core import serializers
def save_results(request):
    if request.method=="POST":
        analyses=BloodAnalysis.objects.filter(user=request.user).order_by('-id')
        analyses_data = list(analyses.values('uploaded_at', 'leukocytes_level', 'hemoglobin_level', 'erythrocytes_level', 'thrombocytes_level', 'hematocrit_level', 'amylase_level', 'potassium_level', 'basophils_level', 'creatinine_level', 'c_reactive_protein_level'))
        context={'analyses':analyses_data}
        return render (request, 'blood_analysis_result.html',context)

def get_analysis_details(request, analysis_id):
    try:
        analysis = BloodAnalysis.objects.get(id=analysis_id)
        data = {
            'leukocytes_level': analysis.leukocytes_level,
            'hemoglobin_level': analysis.hemoglobin_level,
            'erythrocytes_level': analysis.erythrocytes_level,
            'thrombocytes_level': analysis.thrombocytes_level,
            'hematocrit_level': analysis.hematocrit_level,
            'amylase_level': analysis.amylase_level,
            'potassium_level': analysis.potassium_level,
            'basophils_level': analysis.basophils_level,
            'creatinine_level': analysis.creatinine_level,
            'c_reactive_protein_level': analysis.c_reactive_protein_level,
        }
        return JsonResponse(data)
    except BloodAnalysis.DoesNotExist:
        return JsonResponse({'error': 'Analysis not found'}, status=404)

import matplotlib.pyplot as plt
import io
import urllib
from django.shortcuts import render
from django.http import HttpResponse
from .models import BloodAnalysis

import plotly.express as px
import plotly.graph_objects as go  
from django.shortcuts import render
from .models import BloodAnalysis
import plotly.io as pio
import io
import base64
import plotly.graph_objects as go

def show_graph(request, parameter):
    analyses = BloodAnalysis.objects.filter(user=request.user).order_by('-id')  

    analysis_count = len(analyses)
    dates = [f"Analysis {i+1}" for i in range(analysis_count)]

    leukocytes_levels = [analysis.leukocytes_level for analysis in analyses]
    hemoglobin_levels = [analysis.hemoglobin_level for analysis in analyses]
    erythrocytes_levels = [analysis.erythrocytes_level for analysis in analyses]
    thrombocytes_levels = [analysis.thrombocytes_level for analysis in analyses]
    hematocrit_levels = [analysis.hematocrit_level for analysis in analyses]
    amylase_levels = [analysis.amylase_level for analysis in analyses]
    potassium_levels = [analysis.potassium_level for analysis in analyses]
    basophils_levels = [analysis.basophils_level for analysis in analyses]
    creatinine_levels = [analysis.creatinine_level for analysis in analyses]
    c_reactive_protein_levels = [analysis.c_reactive_protein_level for analysis in analyses]

    if parameter == 'leukocytes':
        levels = leukocytes_levels
        title = 'Leukocytes Levels Over Analyses'
    elif parameter == 'erythrocytes':
        levels = erythrocytes_levels
        title = 'Erythrocytes Levels Over Analyses'
    elif parameter == 'thrombocytes':
        levels = thrombocytes_levels
        title = 'Thrombocytes Levels Over Analyses'
    elif parameter == 'hematocrit':
        levels = hematocrit_levels
        title = 'Hematocrit Levels Over Analyses'
    elif parameter == 'hemoglobin':
        levels = hemoglobin_levels
        title = 'Hemoglobin Levels Over Analyses'
    elif parameter == 'amylase':
        levels = amylase_levels
        title = 'Amylase Levels Over Analyses'
    elif parameter == 'potassium':
        levels = potassium_levels
        title = 'Potassium Levels Over Analyses'   
    elif parameter == 'basophils':
        levels = basophils_levels
        title = 'Basophils Levels Over Analyses'
    elif parameter == 'creatinine':
        levels = creatinine_levels
        title = 'Creatinine Levels Over Analyses'
    elif parameter == 'c_reactive_protein':
        levels = c_reactive_protein_levels
        title = 'C-Reactive Protein Levels Over Analyses'
        
        


    levels.reverse()

    fig = px.line(
        x=dates,
        y=levels,
        labels={'value': f'{parameter.capitalize()} Level', 'variable': 'Analysis'},
        title=title
    )

    last_analysis = analyses[0] if analyses.exists() else None  
    recommendations = []

    if last_analysis:
        level = getattr(last_analysis, f'{parameter}_level')
        rec = get_recommendations(parameter, level)
        if rec:
            recommendations.append(rec)

    img_bytes = pio.to_image(fig, format='png')

    graph_image = base64.b64encode(img_bytes).decode('utf-8')

    return render(request, 'blood_analysis_result.html', {
        'graph_image': graph_image,
        'analyses': analyses,
        'selected_parameter': parameter,
        'recommendations': recommendations,  
    })
    
    
def get_recommendations(parameter, level):
    
    recommendations = {
        'leukocytes': {
            'normal': (4.0, 11.0), 
            'low': {
                'message': "У вас низкий уровень лейкоцитов, что может указывать на ослабленную иммунную систему или инфекцию. Рекомендуется проконсультироваться с врачом.",
                'solution': "Для лечения низкого уровня лейкоцитов могут потребоваться лекарства для стимуляции выработки белых кровяных телец. Кроме того, включение в рацион продуктов, богатых витаминами C, E и A, а также цинком, может помочь укрепить вашу иммунную систему. Важно избегать инфекций, пока ваша иммунная система ослаблена."
            },
            'high': {
                'message': "У вас высокий уровень лейкоцитов, что может указывать на инфекцию, воспаление или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для снижения высокого уровня лейкоцитов обычно требуется лечение основной причины, такой как инфекция или воспаление. В зависимости от причины могут быть назначены противовоспалительные препараты или антибиотики. В редких случаях могут потребоваться более специализированные методы лечения заболеваний крови."
            },
        },
        'erythrocytes': {
            'normal': (99, 100), 
            'low': {
                'message': "У вас низкий уровень эритроцитов, что может указывать на анемию. Возможно, вам нужно увеличить потребление продуктов, богатых железом, или принимать железосодержащие добавки.",
                'solution': "Для лечения низкого уровня эритроцитов необходимо увеличить потребление железа. Это можно сделать, употребляя больше красного мяса, листовой зелени, бобовых и обогащенных злаков. Также могут быть рекомендованы добавки железа. Если анемия тяжелая, может потребоваться добавка витамина B12 или фолиевой кислоты."
            },
            'high': {
                'message': "У вас высокий уровень эритроцитов, что может указывать на обезвоживание или другие состояния. Пожалуйста, проконсультируйтесь с медицинским работником.",
                'solution': "Для снижения высокого уровня эритроцитов важно обеспечить правильную гидратацию. Если причиной является обезвоживание, употребление достаточного количества воды должно помочь. При других причинах, таких как полицитемия, врач может рекомендовать флеботомию или лекарства, снижающие выработку красных кровяных телец."
            },
        },
        'thrombocytes': {
            'normal': (150, 450), 
            'low': {
                'message': "У вас низкий уровень тромбоцитов, что может увеличить риск кровотечения. Рекомендуется проконсультироваться с врачом для дальнейшего обследования.",
                'solution': "Низкий уровень тромбоцитов, известный как тромбоцитопения, можно лечить лекарствами, стимулирующими выработку тромбоцитов. В случаях значительного кровотечения может потребоваться переливание тромбоцитов. Также важно избегать лекарств или веществ, которые могут разжижать кровь, таких как аспирин, если это не предписано врачом."
            },
            'high': {
                'message': "У вас высокий уровень тромбоцитов, что может указывать на воспаление, инфекцию или другие состояния. Пожалуйста, обратитесь за медицинской консультацией.",
                'solution': "Для управления высоким уровнем тромбоцитов критически важно устранить основную причину. Если причиной является инфекция, могут потребоваться антибиотики или противовирусные препараты. При таких состояниях, как эссенциальная тромбоцитемия, могут быть назначены лекарства для снижения выработки тромбоцитов или разжижители крови."
            },
        },
        'hematocrit': {
            'normal': (40, 45), 
            'low': {
                'message': "У вас низкий уровень гематокрита, что может указывать на анемию или чрезмерную кровопотерю. Рекомендуется увеличить потребление продуктов, богатых железом и фолиевой кислотой.",
                'solution': "Лечение низкого гематокрита обычно включает устранение причины анемии. Могут потребоваться добавки железа, фолиевой кислоты и инъекции B12 для восстановления нормального уровня. В случаях хронической кровопотери требуется дальнейшее обследование для выявления источника кровотечения."
            },
            'high': {
                'message': "У вас высокий уровень гематокрита, что может указывать на обезвоживание или такие состояния, как полицитемия. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для снижения высокого уровня гематокрита важно восстановить водный баланс. Если причиной является полицитемия, варианты лечения могут включать лекарства для снижения выработки красных кровяных телец или флеботомию (удаление крови). Необходимо контролировать основные состояния."
            },
        },
        'hemoglobin': {
            'normal': (12.0, 17.5),
            'low': {
                'message': "У вас низкий уровень гемоглобина, что может указывать на анемию. Возможно, вам нужно увеличить потребление продуктов, богатых железом, или принимать железосодержащие добавки.",
                'solution': "Для повышения низкого уровня гемоглобина необходимо увеличить потребление железа и рассмотреть возможность приема добавок железа. Продукты, богатые витамином C, могут улучшить усвоение железа. Если анемия вызвана дефицитом витамина B12 или фолиевой кислоты, могут быть назначены соответствующие добавки."
            },
            'high': {
                'message': "У вас высокий уровень гемоглобина, что может указывать на обезвоживание, заболевание легких или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для управления высоким уровнем гемоглобина ключевым является правильная гидратация. Если высокий уровень вызван обезвоживанием, употребление большего количества воды должно помочь. Если это вызвано заболеванием легких или другим состоянием, может потребоваться дальнейшее медицинское вмешательство для лечения основной проблемы."
            },
        },
        'amylase': {
            'normal': (1.0, 1.5),
            'low': {
                'message': "У вас низкий уровень амилазы, что может указывать на недостаточное питание или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для повышения низкого уровня амилазы рекомендуется увеличить потребление продуктов, богатых белками и витаминами. Если это вызвано недостаточным питанием, может потребоваться консультация диетолога."
            },
            'high': {
                'message': "У вас высокий уровень амилазы, что может указывать на инфекцию или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для снижения высокого уровня амилазы важно устранить основную причину. Если причиной является инфекция, могут потребоваться антибиотики или противовирусные препараты. При таких состояниях, как эссенциальная тромбоцитемия, могут быть назначены лекарства для снижения выработки тромбоцитов или разжижители крови."
            },
        },
        'potassium': {
            'normal': (3.5, 5.5),
            'low': {
                'message': "У вас низкий уровень калия, что может указывать на недостаточное питание или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для повышения низкого уровня калия рекомендуется увеличить потребление продуктов, богатых калием, таких как бананы, авокадо, морковь и овощи. Если это вызвано недостаточным питанием, может потребоваться консультация диетолога."
            },
            'high': {
                'message': "У вас высокий уровень калия, что может указывать на недостаточное питание или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для снижения высокого уровня калия важно устранить основную причину. Если причиной является недостаточное питание, может потребоваться консультация диетолога. Если это вызвано другим состоянием, может потребоваться дальнейшее медицинское вмешательство."
            },
        },
        'basophils': {
            'normal': (0.0, 1.0),
            'low': {
                'message': "У вас низкий уровень базофилов, что может указывать на недостаточное питание или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для повышения низкого уровня базофилов рекомендуется увеличить потребление продуктов, богатых белками и витаминами. Если это вызвано недостаточным питанием, может потребоваться консультация диетолога."
            },
            'high': {
                'message': "У вас высокий уровень базофилов, что может указывать на воспаление или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для снижения высокого уровня базофилов важно устранить основную причину. Если причиной является воспаление, могут потребоваться противовоспалительные препараты. Если это вызвано другим состоянием, может потребоваться дальнейшее медицинское вмешательство."
            },
        },
        'creatinine': {
            'normal': (0.5, 1.2),
            'low': {
                'message': "У вас низкий уровень креатинина, что может указывать на недостаточное питание или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для повышения низкого уровня креатинина рекомендуется увеличить потребление продуктов, богатых белками и витаминами. Если это вызвано недостаточным питанием, может потребоваться консультация диетолога."
            },
            'high': {
                'message': "У вас высокий уровень креатинина, что может указывать на недостаточное питание или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для снижения высокого уровня креатинина важно устранить основную причину. Если причиной является недостаточное питание, может потребоваться консультация диетолога. Если это вызвано другим состоянием, может потребоваться дальнейшее медицинское вмешательство."
            },
        },
        'c_reactive_protein': {
            'normal': (0.0, 5.0),
            'low': {
                'message': "У вас низкий уровень ревматоидного фактора, что может указывать на недостаточное питание или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для повышения низкого уровня ревматоидного фактора рекомендуется увеличить потребление продуктов, богатых белками и витаминами. Если это вызвано недостаточным питанием, может потребоваться консультация диетолога."
            },  
            'high': {
                'message': "У вас высокий уровень ревматоидного фактора, что может указывать на воспаление или другие состояния. Пожалуйста, проконсультируйтесь с врачом.",
                'solution': "Для снижения высокого уровня ревматоидного фактора важно устранить основную причину. Если причиной является воспаление, могут потребоваться противовоспалительные препараты. Если это вызвано другим состоянием, может потребоваться дальнейшее медицинское вмешательство."
            },
        },
       
            
        
        
        
    }

    normal_range = recommendations[parameter]['normal']
    try:

        if level < normal_range[0]:
            rec = recommendations[parameter]['low']
            return rec['message'], rec['solution']
        elif level > normal_range[1]:
            rec = recommendations[parameter]['high']
            return rec['message'], rec['solution']
        else:
            return None, None
    except:
        return ['In the last analysis this indicator was not present','']