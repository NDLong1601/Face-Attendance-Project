import cv2
import os

# Tạo thư mục lưu dataset
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Mở webcam
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_id = input('\n Nhập ID (ví dụ 1,2,3,...): ')
print("\n Đang thu thập dữ liệu khuôn mặt. Nhấn 'q' để thoát...")

count = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    
    cv2.imshow('image', img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count >= 50:  # Lưu 50 ảnh cho mỗi người
        break

cam.release()
cv2.destroyAllWindows()
