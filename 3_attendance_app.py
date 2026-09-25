# 3_attendance_app.py
# -*- coding: utf-8 -*-
import time
import os
import cv2
import pandas as pd
from datetime import datetime

# Đọc users từ config.py hoặc file json/csv
try:
    from config import users
except Exception:
    users = {}

# Kiểm tra trainer.yml
if not os.path.exists('trainer.yml'):
    print("Chưa có dữ liệu huấn luyện. Hãy thêm người mới và train trước!")
    exit(1)

# Đọc model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

attendance = pd.DataFrame(columns=['ID', 'Name', 'Time'])
seen_ids = set()
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 70:
            name = users.get(id_pred, f"User{id_pred}")
            if id_pred not in seen_ids:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance = pd.concat([attendance, pd.DataFrame([[id_pred, name, now]], columns=['ID', 'Name', 'Time'])], ignore_index=True)
                seen_ids.add(id_pred)
            cv2.putText(frame, f"{name}", (x+5, y-5), font, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Unknown", (x+5, y-5), font, 1, (0,0,255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
attendance.to_csv('attendance.csv', index=False)
print("Đã lưu danh sách điểm danh vào attendance.csv")
