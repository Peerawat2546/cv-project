import cv2
import numpy as np

min_contour_width = 40  
min_contour_height = 40  
offset = 10  
line_height = 550  
matches = []
vehicles_count = {"car": 0, "motorbike": 0, "truck": 0}

cap = cv2.VideoCapture('Video.mp4')
cap.set(3, 1920)
cap.set(4, 1080)

if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False
ret, frame1 = cap.read()
ret, frame2 = cap.read()

def get_centroid(x, y, w, h):
    return x + w // 2, y + h // 2

def classify_vehicle(w, h):
    if w > 100 and h > 100:
        return "truck"
    elif w > 40 and h > 40:
        return "car"
    else:
        return "motorbike"

while ret:
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= min_contour_width and h >= min_contour_height:
            vehicle_type = classify_vehicle(w, h)
            cx, cy = get_centroid(x, y, w, h)
            matches.append((cx, cy, vehicle_type))
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame1, vehicle_type, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for (x, y, v_type) in matches:
        if line_height - offset < y < line_height + offset:
            vehicles_count[v_type] += 1
            matches.remove((x, y, v_type))
    
    cv2.putText(frame1, f"Cars: {vehicles_count['car']} Motorbikes: {vehicles_count['motorbike']} Trucks: {vehicles_count['truck']}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 0), 2)
    cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)
    cv2.imshow("Vehicle Detection", frame1)
    
    if cv2.waitKey(1) == 27:
        break
    frame1 = frame2
    ret, frame2 = cap.read()

cv2.destroyAllWindows()
cap.release()