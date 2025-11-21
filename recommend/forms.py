# recommend/forms.py
from django import forms

class InputForm(forms.Form):
    N = forms.FloatField(label='Nitrogen (N)', min_value=0, initial=50)
    P = forms.FloatField(label='Phosphorus (P)', min_value=0, initial=30)
    K = forms.FloatField(label='Potassium (K)', min_value=0, initial=30)
    ph = forms.FloatField(label='Soil pH', min_value=0, max_value=14, initial=6.5)
    temperature = forms.FloatField(label='Temperature (°C)', initial=25)
    rainfall = forms.FloatField(label='Rainfall (mm)', initial=100)
    humidity = forms.FloatField(label='Humidity (%)', initial=60) 
