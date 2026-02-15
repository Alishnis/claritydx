from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib import messages
from .models import Analysis, BloodAnalysis, BloodCellAnalysis

def register_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        if User.objects.filter(username=email).exists():
            messages.error(request, 'Пользователь с таким email уже зарегистрирован.')
        else:
            user = User.objects.create_user(username=email, password=password)
            user.save()
            login(request, user)
            return redirect('analysis_page')
    
    return render(request, 'register.html')

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('analysis_page')
        else:
            messages.error(request, 'Неверный email или пароль.')
    return render(request, 'register.html')

from .models import AnalysisCT, AnalysisSkin, Analysis  
from django.core import serializers
def user_kab(request):
    """Показать личный кабинет с разным контентом в зависимости от выбранного раздела."""
    selected_section = request.GET.get('section', 'ct')  

    context = {
        'user': request.user,
    }

    if selected_section == 'ct':
        analyses = AnalysisCT.objects.filter(user=request.user).order_by('-id')
        context.update({
            'section_title': 'Анализ КТ изображений',
            'analyses': analyses,
        })
    elif selected_section == 'xray':
        analyses = Analysis.objects.filter(user=request.user).order_by('-id')  
        context.update({
            'section_title': 'Анализ рентгеновских снимков',
            'analyses': analyses,
        })
    elif selected_section == 'skin':
        analyses = AnalysisSkin.objects.filter(user=request.user).order_by('-id')  
        context.update({
            'section_title': 'Анализ кожи',
            'analyses': analyses,
        })
    elif selected_section == 'blood':
        analyses = BloodAnalysis.objects.filter(user=request.user).order_by('-id')
        analyses_data = list(analyses.values('uploaded_at', 'leukocytes_level', 'hemoglobin_level', 'erythrocytes_level', 'thrombocytes_level', 'hematocrit_level'))
        return render(request, 'blood_analysis_result.html', {'analyses': analyses_data})
    elif selected_section == 'blood_cell':
        analyses = BloodCellAnalysis.objects.filter(user=request.user).order_by('-id')
        context.update({
            'section_title': 'Анализ клеток крови',
            'analyses': analyses,
        })
    else:
        context.update({
            'section_title': 'Неизвестный раздел',
            'analyses': [],
        })

    return render(request, 'user_kab.html', context)