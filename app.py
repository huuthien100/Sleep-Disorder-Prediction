from flask import Flask, render_template, request
import joblib
import time
import csv
from datetime import datetime

app = Flask(__name__)

# Load mô hình RandomForest từ file
best_random_forest_model = joblib.load("best_random_forest_model.joblib")

# Load các encoder từ file
gender_encoder = joblib.load("gender_encoder.joblib")
occupation_encoder = joblib.load("occupation_encoder.joblib")
bmi_category_encoder = joblib.load("bmi_category_encoder.joblib")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        # Nhận các giá trị từ form
        gender = request.form['gender']
        age = int(request.form['age'])
        occupation = request.form['occupation']
        sleep_duration = float(request.form['sleep_duration'])
        quality_of_sleep = int(request.form['quality_of_sleep'])
        physical_activity_level = int(request.form['physical_activity_level'])
        stress_level = int(request.form['stress_level'])
        bmi_category = request.form['bmi_category']
        
        # Xử lý blood_pressure thành systolic và diastolic
        systolic, diastolic = map(int, request.form['blood_pressure'].split('/'))

        heart_rate = int(request.form['heart_rate'])
        daily_steps = int(request.form['daily_steps'])

        # Encode các giá trị hạng mục
        gender_encoded = gender_encoder.transform([gender])[0]
        occupation_encoded = occupation_encoder.transform([occupation])[0]
        bmi_category_encoded = bmi_category_encoder.transform([bmi_category])[0]

        # Chuyển đổi các đặc trưng thành một mảng hoặc dataframe
        features = [[gender_encoded,age , occupation_encoded, sleep_duration, 
                     quality_of_sleep, physical_activity_level, stress_level, 
                     bmi_category_encoded, heart_rate, daily_steps,systolic, diastolic]]

        # Thực hiện dự đoán bằng cách sử dụng mô hình RandomForest đã được nạp
        prediction_result = best_random_forest_model.predict(features)
        time.sleep(2)
        # Chuyển đổi kết quả dự đoán thành chuỗi tương ứng
        if prediction_result[0] == 2:
            prediction_str = "None"
        elif prediction_result[0] == 1:
            prediction_str = "Sleep Apnea"
        else:
            prediction_str = "Insomnia"
        # Lưu thông tin vào dataset
        with open('user_data.csv', 'a', newline='') as csvfile:
            fieldnames = ['Timestamp', 'Gender', 'Age', 'Occupation', 'SleepDuration', 
                           'QualityOfSleep', 'PhysicalActivityLevel', 'StressLevel', 
                           'BMICategory', 'BloodPressure', 'HeartRate', 'DailySteps', 'SleepDisorder']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Thực hiện gộp 'Systolic' và 'Diastolic'
            blood_pressure_str = f"{systolic}/{diastolic}"

            # Ghi dữ liệu mới vào CSV
            writer.writerow({
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Gender': gender,
                'Age': age,
                'Occupation': occupation,
                'SleepDuration': sleep_duration,
                'QualityOfSleep': quality_of_sleep,
                'PhysicalActivityLevel': physical_activity_level,
                'StressLevel': stress_level,
                'BMICategory': bmi_category,
                'BloodPressure': blood_pressure_str,
                'HeartRate': heart_rate,
                'DailySteps': daily_steps,
                'SleepDisorder': prediction_str
            })
        # Chuyển hướng sang trang kết quả và truyền kết quả dự đoán
        return render_template('result.html', prediction_result=prediction_str)

if __name__ == '__main__':
    app.run(debug=True)
