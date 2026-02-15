from django.contrib import admin
from .models import Analysis,Disease,AnalysisCT,AnalysisSkin,BloodAnalysis

# Настройка названия сайта в админке
admin.site.site_header = "HealthX Admin"
admin.site.site_title = "HealthX Admin"
admin.site.index_title = "Администрирование HealthX"

admin.site.register(Analysis)
admin.site.register(Disease)
admin.site.register(AnalysisCT)
admin.site.register(AnalysisSkin)
admin.site.register(BloodAnalysis)
