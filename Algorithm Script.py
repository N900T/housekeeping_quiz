import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

# Create an instance of the OneHotEncoder class
ohe = OneHotEncoder()

#get pandas to read from Google Sheets
sheet_id = "11Q6_0mnEDvFkgj8sS1Y_Cy23WJn3-kXR-k-nWZG9GBw"
sheet_name = 'Response'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
df_responses = pd.read_csv(url)

df_impersonations = df_responses['Type in which housekeeping item you are impersonating']
df_responses = df_responses.drop(
    ['Type in which housekeeping item you are impersonating', 'Timestamp'], axis=1
    )

#preprocessing df_responses
response_array = ohe.fit_transform(df_responses).toarray()
response_labels = ohe.get_feature_names_out()
response_labels = response_labels.ravel()

df_responses_preprocessed = pd.DataFrame(
    response_array, columns=response_labels
    )

#set up data
y = df_impersonations
x = df_responses_preprocessed

#fit model
SVM_model = SVC(gamma='auto')
SVM_model.fit(x, y)


#making testing data to predict stuff
ohe_test = OneHotEncoder()

sheet_id_test = "11Q6_0mnEDvFkgj8sS1Y_Cy23WJn3-kXR-k-nWZG9GBw"
sheet_name_test = 'Response'
url_test = f'https://docs.google.com/spreadsheets/d/{sheet_id_test}/gviz/tq?tqx=out:csv&sheet={sheet_name_test}'
df_responses_test = pd.read_csv(url_test)

df_impersonations_test = df_responses_test['Type in which housekeeping item you are impersonating']
df_responses_test = df_responses_test.drop(
    ['Type in which housekeeping item you are impersonating', 'Timestamp'], axis=1
    )

response_array_test = ohe_test.fit_transform(df_responses_test).toarray()
response_labels_test = ohe_test.get_feature_names_out()
response_labels_test = response_labels_test.ravel()

df_responses_preprocessed_test = pd.DataFrame(
    response_array_test, columns=response_labels_test
    )

SVM_model.predict(df_responses_preprocessed_test)