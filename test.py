import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
# Example Data user from X_test (index 0) with 43 column
data_user = np.array([[ 3.8746e+02,  6.2920e+01,  4.0100e+00,  3.2300e+00,  5.2000e-01,
         4.3300e+00, -5.0000e-01,  3.9300e+00,  1.2344e+02,  7.3000e-01,
         1.2450e+01,  8.4280e+01,  1.3436e+02,  1.1796e+02,  4.4000e-01,
         9.2000e-01, -3.5000e-01,  4.0400e+00, -4.9000e-01, -1.0000e-01,
         9.4000e-01, -4.0000e-02, -3.4000e-01,  3.7000e-01,  5.2900e+00,
         1.5190e+01,  2.8000e-01,  1.6486e+02,  4.6100e+01,  4.6860e+01,
        -7.9000e-01,  7.3100e+00, -3.1000e-01, -3.4000e-01,  2.4000e-01,
        -3.9000e-01,  3.7000e-01,  5.1150e+01,  1.0600e+00,  1.0800e+00,
         1.6000e-01, -6.4000e-01,  1.1850e+01]])
dataset=pd.read_csv('List_Label.csv')
model = load_model(r'Models\food_model_1.h5')
y_pred = model.predict(data_user)
top_5 = np.argsort(y_pred.flatten())[-5:]
selected_rows = dataset.loc[dataset['label'].isin(top_5), ['name', 'label']]
selected_rows.sort_values(by='label',ascending=True,inplace=True)
# Print or use the selected rows
print(selected_rows.drop_duplicates(subset=['label']))