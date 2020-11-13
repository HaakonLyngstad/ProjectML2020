from django.shortcuts import render
from django.http import HttpResponse
from mlwebsite import settings


# Create your views here.


def main_view(request):
    variable = 3
    return render(request, 'main.html')
    #return HttpResponse('Hællæ på ræ')
