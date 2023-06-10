from flask import Flask, render_template, request,session, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from flask_session import Session
import sqlite3



app = Flask(__name__,static_folder='public')
sess = Session()

#Create database 
db_path='patients.db'

def create_table():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patientName TEXT,
            male INTEGER,
            age INTEGER,
            education INTEGER,
            currentSmoker INTEGER,
            cigsPerDay INTEGER,
            BPMeds INTEGER,
            prevalentStroke INTEGER,
            prevalentHyp INTEGER,
            diabetes INTEGER,
            totChol INTEGER,
            sysBP INTEGER,
            diaBP INTEGER,
            BMI REAL,
            heartRate INTEGER,
            glucose INTEGER,
            isCAD INTEGER
        )
    ''')
    conn.commit()
    conn.close()
#loading the model using joblib
try:
    model = joblib.load('best_model_.pkl')
    print('Model loaded successfully using joblib!')
except Exception as e:
    print('Error loading model using joblib:', e)



@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check the username and password
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/index')
def form():
    return render_template('index.html')


def predict_cad(male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP ,diaBP, BMI, heartRate, glucose):

    array_features = [np.array([male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose])]

    # Example feature names
    feature_names = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                     'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
                     'diaBP', 'BMI', 'heartRate', 'glucose']

    # Create DataFrame from array with column names
    input_data = pd.DataFrame(data=array_features, columns=feature_names)
    prediction = model.predict(input_data)
    output = prediction

    # Check the output values and retrieve the result with html tag based on the value
    return output

@app.route('/dashboard')
def dashboard():
    #Connect Database
    conn=sqlite3.connect(db_path)
    c=conn.cursor()
    #Query data from database
    c.execute('Select * from patients')
    data=c.fetchall()
    conn.close()
    return render_template('dashboard.html', data=data)
        
@app.route('/Predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    patientName = request.form['patientName']
    male = int(request.form['male'])
    age = int(request.form['age'])
    education = int(request.form['education'])
    currentSmoker = int(request.form['currentSmoker'])
    cigsPerDay = int(request.form['cigsPerDay'])
    BPMeds = int(request.form['BPMeds'])
    prevalentStroke = int(request.form['prevalentStroke'])
    prevalentHyp = int(request.form['prevalentHyp'])
    diabetes = float(request.form['diabetes'])
    totChol = int(request.form['totChol'])
    sysBP = float(request.form['sysBP'])
    diaBP = float(request.form['diaBP'])
    BMI = float(request.form['BMI'])
    heartRate = int(request.form['heartRate'])
    glucose = int(request.form['glucose'])
    
    
    #Make the prediction
    prediction = predict_cad(male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)
    print(prediction)
    if prediction==0:
        isCAD=0
        kq="This patient does not have coronary artery disease"
    else:
        kq="This patient does have coronary artery disease"
        isCAD=1
  
    # Kết nối cơ sở dữ liệu SQLite
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Thêm thông tin bệnh nhân vào bảng patients
    c.execute('''
    INSERT INTO patients (patientName, male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose, isCAD)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (patientName, male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose, isCAD))

    conn.commit()
    conn.close()
   

    return render_template('results.html', prediction=kq)

@app.route('/delete-all',methods=['POST'])
def deleteAll():
    conn=sqlite3.connect(db_path)
    c=conn.cursor()
    c.execute('Delete from patients')
    conn.commit()
    conn.close
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    create_table()
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    sess.init_app(app)
    app.debug = True
    app.run()
    

