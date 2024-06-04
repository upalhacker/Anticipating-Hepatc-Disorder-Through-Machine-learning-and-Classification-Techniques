import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the dataset
liver = pd.read_csv('C:\sai\indian_liver_patient.csv')

# Preprocess the data
le = LabelEncoder()
liver['Gender'] = le.fit_transform(liver['Gender'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
liver = pd.DataFrame(imputer.fit_transform(liver), columns=liver.columns)

# Drop rows with missing values in the target variable
liver.dropna(subset=['Dataset'], inplace=True)

# Split the data into features and target
X = liver.drop(['Dataset', 'Direct_Bilirubin'], axis=1)
y = liver['Dataset'].replace({'Yes': 1, 'No': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Get the list of feature names
feature_names = X.columns.tolist()

# Fit the LogisticRegression model on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model using pickle
filename = 'sai.pkl'
pickle.dump(model, open(filename, 'wb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model = pickle.load(open('sai.pkl', 'rb'))

    # Get the input values from the form
    age = float(request.form['Age'])
    gender = int(request.form['Gender'])
    total_bilirubin = float(request.form['Total_Bilirubin'])
    alkaline_phosphotase = float(request.form['Alkaline_Phosphotase'])
    alamine_aminotransferase = float(request.form['Alamine_Aminotransferase'])
    aspartate_aminotransferase = float(request.form['Aspartate_Aminotransferase'])
    total_protiens = float(request.form['Total_Protiens'])
    albumin = float(request.form['Albumin'])
    albumin_globulin_ratio = float(request.form['Albumin_and_Globulin_Ratio'])

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Total_Bilirubin': [total_bilirubin],
        'Alkaline_Phosphotase': [alkaline_phosphotase],
        'Alamine_Aminotransferase': [alamine_aminotransferase],
        'Aspartate_Aminotransferase': [aspartate_aminotransferase],
        'Total_Protiens': [total_protiens],
        'Albumin': [albumin],
        'Albumin_and_Globulin_Ratio': [albumin_globulin_ratio]
    })

    # Make predictions using the loaded model
    prediction = model.predict(input_data)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
