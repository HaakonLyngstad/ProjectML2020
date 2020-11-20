from django.shortcuts import render
from django.http import HttpResponseRedirect
from mlwebsite import settings
from .forms import DatasetForm

# Create your views here.


def main_view(request):
    return render(request, 'main.html')

def methods(request):
    return render(request, 'methods.html')

def support(request):
    print(request.FILES)
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            # file is saved
            form.save()
            return HttpResponseRedirect('/support')
    else:
        form = DatasetForm()
    return render(request, 'support.html', {
        'form': form,
        })


def bagging_view(self):
    return render(request, 'bagging.html')

def boosting_view(self):
    return render(request, 'boosting.html')

def stacking_view(self):
    return render(reequest, 'stacking.html')

