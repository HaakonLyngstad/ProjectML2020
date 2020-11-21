from django import forms
from .models import Dataset
import pandas as pd


class DatasetForm(forms.ModelForm):

    def clean(self):
        data = self.cleaned_data
        print(data.get("data_file"))
        if not (data.get("data_file")):

            raise forms.ValidationError('You can only upload csv files to this form')
        else:
            try:
                pd.read_csv(data.get("data_file"), usecols=["text", "fake"])
            except Exception as e:
                print(e)
                raise forms.ValidationError('The csv file does not have the right format')
            else:
                return self.cleaned_data

    class Meta:
        model = Dataset
        fields = ['title', 'comment', 'data_file']


class PredictForm(forms.Form):
    text = forms.CharField(label='Prediction text', widget=forms.Textarea, max_length=10000)
