from django.shortcuts import render
from django.http import HttpResponseRedirect, Http404
from mlwebsite import settings
from .forms import DatasetForm, PredictForm
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import pickle
from django.contrib.staticfiles.storage import staticfiles_storage
#from django.utils import simplejson
# Create your views here.


def predicate_text(text):

    model_filename = staticfiles_storage.path('models/SVC.pickle')
    with open(model_filename, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    tfidf_filename =  staticfiles_storage.path("models/ngram_vectorizer.pickle")
    with open(tfidf_filename, 'rb') as tfidf_file:
        loaded_tfidf = pickle.load(tfidf_file)

    text = loaded_tfidf.transform([text])
    result = loaded_model.predict(text)

    return result[0].item()


def main_view(request):
    if request.method == 'POST':
        form = PredictForm(request.POST, request.FILES)
        if form.is_valid():
            text = form.data['text']
            # file is saved
            print(text)
            result = predicate_text(text)
            return render(request, 'main.html', {'form': result, "show_form": False, "result": result})
    else:
        form = PredictForm()
    return render(request, 'main.html', {'form': form, 'show_form': True})


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
        body = json.loads(request.body)
        text = body["text"]
        result = predicate_text(text=text)
        return JsonResponse({"fradulent": result})
    else:
        raise Http404()


def bagging_view(request):
    return render(request, 'bagging.html')


def boosting_view(request):
    return render(request, 'boosting.html')


def stacking_view(request):
    return render(request, 'stacking.html')


def classifier_view(request):
    return render(request, 'classifiers.html')
