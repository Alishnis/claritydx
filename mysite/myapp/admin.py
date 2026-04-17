from django.contrib import admin
from .models import Analysis,Disease,AnalysisCT,AnalysisSkin,BloodAnalysis

# Настройка названия сайта в админке
admin.site.site_header = "ClarityDX Admin"
admin.site.site_title = "ClarityDX Admin"
admin.site.index_title = "Администрирование ClarityDX"

admin.site.register(Analysis)
admin.site.register(Disease)
admin.site.register(AnalysisCT)
admin.site.register(AnalysisSkin)
admin.site.register(BloodAnalysis)
