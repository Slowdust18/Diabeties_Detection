from django import forms

class PatientFeatureForm(forms.Form):
    # Binary Features
    HighBp = forms.ChoiceField(label="High Blood Pressure", choices=[(1, 'Yes'), (0, 'No')], widget=forms.RadioSelect)
    Highchol = forms.ChoiceField(label="High Cholesterol", choices=[(1, 'Yes'), (0, 'No')], widget=forms.RadioSelect)
    HeartDiseaseorAttack = forms.ChoiceField(label="Heart Disease or Attack", choices=[(1, 'Yes'), (0, 'No')], widget=forms.RadioSelect)
    Stroke = forms.ChoiceField(label="Stroke", choices=[(1, 'Yes'), (0, 'No')], widget=forms.RadioSelect)
    Smoker = forms.ChoiceField(label="Smoker", choices=[(1, 'Yes'), (0, 'No')], widget=forms.RadioSelect)
    PhysActivity = forms.ChoiceField(label="Physical Activity", choices=[(1, 'Yes'), (0, 'No')], widget=forms.RadioSelect)
    DiffWalk = forms.ChoiceField(label="Difficulty Walking", choices=[(1, 'Yes'), (0, 'No')], widget=forms.RadioSelect)
    Sex = forms.ChoiceField(label="Sex", choices=[(1, 'Male'), (0, 'Female')], widget=forms.RadioSelect)
    HvyAlcoholConsump = forms.ChoiceField(label="Heavy Alcohol Consumption", choices=[(1, 'Yes'), (0, 'No')], widget=forms.RadioSelect)

    # Range Features
    GenHlth = forms.ChoiceField(label="General Health", choices=[
        (1, 'Excellent'), (2, 'Very Good'), (3, 'Good'), (4, 'Fair'), (5, 'Poor')
    ])
    MentHlth = forms.IntegerField(label="Mental Health (bad days last 30)", min_value=0, max_value=30)
    PhysHlth = forms.IntegerField(label="Physical Health (bad days last 30)", min_value=0, max_value=30)

    Age = forms.ChoiceField(label="Age Group", choices=[
        (1, '18–24'), (2, '25–29'), (3, '30–34'), (4, '35–39'),
        (5, '40–44'), (6, '45–49'), (7, '50–54'), (8, '55–59'),
        (9, '60–64'), (10, '65–69'), (11, '70–74'), (12, '75–79'), (13, '80 or older')
    ])

    # Numeric Feature
    Bmi = forms.FloatField(label="BMI", min_value=10, max_value=60)
