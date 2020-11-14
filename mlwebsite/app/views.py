from django.shortcuts import render
from django.http import HttpResponseRedirect
from mlwebsite import settings
from .forms import DatasetForm

# Create your views here.


def main_view(request):
    variable = 3
    return render(request, 'main.html')
    #return HttpResponse('Hællæ på ræ')


def predicate(request):
    return render(request, 'predicate.html')


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
    return render(request, 'support.html', {'form': form})