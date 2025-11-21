# recommend/api.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .predictor import predict

@api_view(['POST'])
def api_recommend(request):
    data = request.data
    # encode season if present
    season_map = {'Kharif':1,'Rabi':2,'Zaid':3}
    if 'season' in data:
        try:
            data['season_encoded'] = season_map.get(data.pop('season'))
        except Exception:
            pass
    try:
        result = predict(data)
    except Exception as e:
        return Response({'error': str(e)}, status=400)
    return Response(result)
