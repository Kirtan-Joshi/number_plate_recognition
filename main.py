import cv2
import numpy as np
import pytesseract
import time
import imutils

# raspberrypi pi parthishere
# parthgoodboy11

def detect_text(test_license_plate):
  resize_test_license_plate = cv2.resize(test_license_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
  grayscale_resize_test_license_plate = cv2.cvtColor(resize_test_license_plate, cv2.COLOR_BGR2GRAY)

  gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (5, 5), 0)

  new_predicted_result_GWT2180 = pytesseract.image_to_string(gaussian_blur_license_plate, lang='eng',
                                                             config='--oem 3 -l eng --psm 6 ')
  return new_predicted_result_GWT2180

plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
# video = cv2.VideoCapture("Downloads/vid.mp4")
video = cv2.VideoCapture(0)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
custom_config = r'--oem 3 --psm 6'

if(video.isOpened()==False):
    print('Error Reading Video')

while True:
  time.sleep(0.1)
  ret, frame = video.read()
  gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



  plate = plat_detector.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))
  for (x, y, w, h) in plate:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    frame = frame[y:y+h, x:x+w]
    # frame[y:y + h, x:x + w] = cv2.blur(frame[y:y + h, x:x + w], ksize=(10, 10))

    # frame = cv2.bilateralFilter(frame, 11, 17, 17)
    # frame = cv2.Canny(frame, 30, 200)
    # cnts, new = cv2.findContours(frame.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow('Video', edged)



    # text = pytesseract.image_to_string(frame, config=custom_config)

    text = detect_text(frame)

    print("text")
    print(text)

    cv2.putText(frame, text='License Plate', org=(x - 3, y - 3), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255),
                thickness=1, fontScale=0.6)

    cv2.putText(frame, text=text, org=(x-5, y-5), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), fontScale=0.5, thickness=1)

  if ret == True:
    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
      break
  else:
    break

video.release()
cv2.destroyAllWindows()


