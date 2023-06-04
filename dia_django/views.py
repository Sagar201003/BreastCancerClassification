from django.shortcuts import render,HttpResponse,redirect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from pycaret.datasets import get_data
from pycaret.classification import *

def view(request):
    return render(request, 'index.html')
def get_user_input(request):
# Load and preprocess the BreastCancer dataset
    data = pd.read_csv('BreastCancerdata.csv')
# preprocess the dataset as needed
    X = np.array(data.drop(['diagnosis', 'id'], axis=1))
    Y = np.array(data.diagnosis)
# Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=2)
# Train a  model
    from sklearn.ensemble import AdaBoostClassifier
    adb=AdaBoostClassifier(random_state=97)
    adb.fit(X_train,Y_train)
    adb.score(X_test,Y_test)

    exp = setup(data= data, target='diagnosis',train_size=0.7, session_id=123)
    best_model = compare_models(fold=4)
    # check the final params of best model
    best_model.get_params()
    evaluate_model(best_model)

    tuned_model = tune_model(best_model)
    predictions = predict_model(tuned_model, data=data.drop('diagnosis', axis = 1, inplace = True))
    tuned_model.fit(X_train,Y_train)

    tuned_model.score(X_train,Y_train)

    tuned_model.score(X_test,Y_test)

# Define a function to take user input

    if request.method == "POST":
    
        id = int(request.POST.get('id', 0))
        radius_mean = float(request.POST.get('radius_mean', 0))
        texture_mean = float(request.POST.get('texture_mean', 0))
        perimeter_mean = float(request.POST.get('perimeter_mean', 0))
        area_mean = float(request.POST.get('area_mean', 0))
        smoothness_mean = float(request.POST.get('smoothness_mean', 0))
        compactness_mean = float(request.POST.get('compactness_mean', 0))
        concavity_mean = float(request.POST.get('concavity_mean', 0))
        concave_points_mean = float(request.POST.get('concave_points_mean', 0))
        symmetry_mean = float(request.POST.get('symmetry_mean', 0))
        fractal_dimension_mean = float(request.POST.get('fractal_dimension_mean', 0))
        radius_se = float(request.POST.get('radius_se', 0))
        texture_se = float(request.POST.get('texture_se', 0))
        perimeter_se = float(request.POST.get('perimeter_se', 0))
        area_se = float(request.POST.get('area_se', 0))
        smoothness_se = float(request.POST.get('smoothness_se', 0))
        compactness_se = float(request.POST.get('compactness_se', 0))
        concavity_se = float(request.POST.get('concavity_se', 0))
        concave_points_se = float(request.POST.get('concave_points_se', 0))
        symmetry_se = float(request.POST.get('symmetry_se', 0))
        fractal_dimension_se = float(request.POST.get('fractal_dimension_se', 0))
        radius_worst = float(request.POST.get('radius_worst', 0))
        texture_worst = float(request.POST.get('texture_worst', 0))
        perimeter_worst = float(request.POST.get('perimeter_worst', 0))
        area_worst = float(request.POST.get('area_worst', 0))
        smoothness_worst = float(request.POST.get('smoothness_worst', 0))
        compactness_worst = float(request.POST.get('compactness_worst', 0))
        concavity_worst = float(request.POST.get('concavity_worst', 0))
        concave_points_worst = float(request.POST.get('concave_points_worst', 0))
        symmetry_worst = float(request.POST.get('symmetry_worst', 0))
        fractal_dimension_worst = float(request.POST.get('fractal_dimension_worst', 0))





        
 
# Call your input function to get user input
    user_input =  np.array([radius_mean, texture_mean, perimeter_mean, area_mean,smoothness_mean, compactness_mean, concavity_mean,
                        concave_points_mean, symmetry_mean, fractal_dimension_mean,
                        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
                        compactness_se, concavity_se, concave_points_se, symmetry_se,
                        fractal_dimension_se, radius_worst, texture_worst,
                        perimeter_worst, area_worst, smoothness_worst,
                        compactness_worst, concavity_worst, concave_points_worst,
                        symmetry_worst, fractal_dimension_worst])
    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(user_input)

    # reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = tuned_model.predict(input_data_reshaped)
    if prediction == [0]:
        # return HttpResponse("The predicted outcome : You have Benign Breast Cancer")
        return render(request, 'benign.html')
    else:
        # return HttpResponse("The predicted outcome : You have Malignant Breast Cancer") 
        return render(request, 'malignant.html')
    
    
    

    



# Create your views here.
