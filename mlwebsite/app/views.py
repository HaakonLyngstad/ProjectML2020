from django.shortcuts import render
from django.http import HttpResponse
from mlwebsite import settings

# Create your views here.


def main_view(self):
    #return render(request, 'main.hmtl')
    return HttpResponse('Hællæ på ræ')
