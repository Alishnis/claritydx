from django.urls import path
from . import views,auth_views,views2,views3_for_models
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('analysis/', views.analysis_page, name='analysis_page'),
    path('login/',auth_views.login_view,name='login'),
    path('register/', auth_views.register_view, name='register'),
    path('kab/',auth_views.user_kab,name='user_kab'),
    path('uploadfila/',views.handle_upload,name="handle_upload"),
    
   
    
    path('analyze/', views.analyze_symptoms, name='analyze_symptoms'),
    path('recovery/', views.treatment_view, name='treatment_view'),
    path('upload/',views.analyze_image,name='analyze_image'),
    path('save_results/', views.save_results, name='save_results'),
    path('upload2/', views.analyze_image2, name='analyze_image2'),
   
    path("skin/",views2.analyze_skin_image,name='analyze_skin_image'),
    path('save_results2/', views2.save_results_skin, name='save_results_skin'),
    path('save_results3/', views.save_results_ct, name='save_results_ct'),
    path('bloodanalysis/',views.process_blood_analysis_file, name='blood_analysis'),
    path('get_analysis_details/<int:analysis_id>/', views2.get_analysis_details, name='get_analysis_details'),

    path('blood_analysis_results/<str:parameter>/', views2.show_graph, name='show_graph'),
  
    path('upload_cellblood/', views3_for_models.upload_image, name='upload_image'),
    
    path('recommendation/', views3_for_models.recommendation_view, name='recommendation'),
    
    path('get_ai_recommendations/', views.get_ai_recommendations, name='get_ai_recommendations'),
    
    path('chatbot_api/', views.chat_with_openai, name='chatbot_api'),
    
   
    path('subscription/', views.subscription_view, name='subscription'),
    path('create-checkout-session/', views.create_checkout_session, name='create_checkout_session'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
