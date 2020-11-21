from django.shortcuts import render
from django.http import HttpResponseRedirect, Http404
from mlwebsite import settings
from .forms import DatasetForm, PredictForm
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

#from django.utils import simplejson
# Create your views here.


def main_view(request):
    if request.method == 'POST':
        form = PredictForm(request.POST, request.FILES)
        if form.is_valid():
            # file is saved
            return HttpResponseRedirect('/')
    else:
        form = PredictForm()
    return render(request, 'main.html', {'form': form})


def predicate(request):
    return render(request, 'predicate.html')

def methods(request):
    return render(request, 'methods.html')

def support(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            # file is saved
            form.save()
            return HttpResponseRedirect('/support')
    else:
        form = DatasetForm()
    return render(request, 'support.html', {'form': form})


@csrf_exempt
def api_predicate(request):
    if request.method == 'POST':
        # last model 
        # test test
        return JsonResponse({"fradulent": 0})
    else:
        return Http404()


def bagging_view(request):
    return render(request, 'bagging.html')


def boosting_view(request):
    return render(request, 'boosting.html')


def stacking_view(request):
    return render(request, 'stacking.html')

