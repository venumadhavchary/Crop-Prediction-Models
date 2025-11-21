# recommend/views.py
from django.shortcuts import render
from .forms import InputForm
from .predictor import predict


def home(request):
    info = {}
    try:
        info = get_model_info()
    except Exception as e:
        info = {'error': str(e)}

    if request.method == 'POST':
        form = InputForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data 
            try:
                result = predict(data)
            except Exception as e:
                result = {'error': str(e)}
            return render(request, 'recommend/result.html', {'result': result, 'input': data, 'model_info': info})
    else:
        form = InputForm()
    return render(request, 'recommend/home.html', {'form': form, 'model_info': info})
