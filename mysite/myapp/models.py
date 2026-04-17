from django.db import models

from django.contrib.auth.models import User
from django.core.exceptions import SuspiciousFileOperation


class Analysis(models.Model):
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    analysis_file = models.FileField(upload_to='uploads/analysis/',null=True)
    gradcam_path = models.TextField(blank=True, null=True)
    result = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    @property
    def analysis_file_exists(self):
        try:
            return bool(self.analysis_file and self.analysis_file.name and self.analysis_file.storage.exists(self.analysis_file.name))
        except (ValueError, SuspiciousFileOperation):
            return False

class AnalysisCT(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    analysis_file = models.FileField(upload_to='uploads/analysis/',null=True)
    gradcam_path = models.TextField(blank=True, null=True)
    result = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    @property
    def analysis_file_exists(self):
        try:
            return bool(self.analysis_file and self.analysis_file.name and self.analysis_file.storage.exists(self.analysis_file.name))
        except (ValueError, SuspiciousFileOperation):
            return False

class AnalysisSkin(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    analysis_file = models.FileField(upload_to='uploads/analysis/',null=True)
    gradcam_path = models.TextField(blank=True, null=True)
    result = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    @property
    def analysis_file_exists(self):
        try:
            return bool(self.analysis_file and self.analysis_file.name and self.analysis_file.storage.exists(self.analysis_file.name))
        except (ValueError, SuspiciousFileOperation):
            return False
    
class Disease(models.Model):
    name = models.CharField(max_length=200)
    
    
    treatment = models.TextField()
    
    

    def __str__(self):
        return self.name
    
class CTAnalysis(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    ct_file = models.FileField(upload_to='uploads/ct_analysis/', null=True)
    result = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CT Analysis by {self.user.username} on {self.uploaded_at}"

from django.db import models
from django.contrib.auth.models import User

class BloodAnalysis(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    analysis_file = models.FileField(upload_to='uploads/analysis/', null=True)
    leukocytes_level = models.FloatField(null=True, blank=True) 
    hemoglobin_level = models.FloatField(null=True, blank=True)  
    erythrocytes_level = models.FloatField(null=True, blank=True)   
    thrombocytes_level=models.FloatField(null=True,blank=True)
    hematocrit_level=models.FloatField(null=True, blank=True)
    amylase_level = models.FloatField(null=True, blank=True)
    potassium_level = models.FloatField(null=True, blank=True)
    basophils_level = models.FloatField(null=True, blank=True)  # New parameter
    creatinine_level = models.FloatField(null=True, blank=True)  # New parameter
    c_reactive_protein_level = models.FloatField(null=True, blank=True)
    
    uploaded_at = models.DateTimeField(auto_now_add=True)

    @property
    def analysis_file_exists(self):
        try:
            return bool(self.analysis_file and self.analysis_file.name and self.analysis_file.storage.exists(self.analysis_file.name))
        except (ValueError, SuspiciousFileOperation):
            return False

    def __str__(self):
        return f"Blood Analysis by {self.user.username} on {self.uploaded_at}"

class BloodCellAnalysis(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    image = models.ImageField(upload_to='blood_cell_images/')
    prediction = models.CharField(max_length=100)
    confidence = models.FloatField()
    saliency_map = models.ImageField(upload_to='saliency_maps/')
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def image_exists(self):
        try:
            return bool(self.image and self.image.name and self.image.storage.exists(self.image.name))
        except (ValueError, SuspiciousFileOperation):
            return False

    @property
    def saliency_map_exists(self):
        try:
            return bool(self.saliency_map and self.saliency_map.name and self.saliency_map.storage.exists(self.saliency_map.name))
        except (ValueError, SuspiciousFileOperation):
            return False
    
    def __str__(self):
        return f"{self.prediction} - {self.created_at}"
