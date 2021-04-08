import RPi.GPIO as GPIO
from RPLCD import CharLCD

from smbus2 import SMBus
from mlx90614 import MLX90614

import time

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#from imutils.video import VideoStream
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import imutils
import cv2
import os


conv_belt= 36
sanitizer = 15
pulse_start= 0
pulse_end = 0
TRIG = 18
ECHO = 16
sanitize_delay = 3
door_open = 22  
door_close = 21

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)
GPIO.setup(conv_belt,GPIO.OUT)
GPIO.setup(sanitizer,GPIO.OUT)
GPIO.setup(door_open, GPIO.OUT)
GPIO.setup(door_close, GPIO.OUT)
lcd = CharLCD(cols=16, rows=2, pin_rs=37, pin_e=35, pins_data=[33, 31, 29, 23],numbering_mode=GPIO.BOARD)

GPIO.output(door_open,True)
GPIO.output(door_close,True)
GPIO.output(sanitizer,True)
GPIO.output(conv_belt,True)


def check_dist():
    pulse_start= 0
    pulse_end = 0
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO)==0:
        pulse_start = time.time()

    while GPIO.input(ECHO)==1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance+1.15,2)
    #print(distance)
    return distance

def check_temp():
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    #print ("Ambient Temperature :", sensor.get_ambient())
    temp = sensor.get_object_1()
    #print ("Object Temperature :", temp)
    bus.close()
    temp= (9/5 * temp)+32
    #print(temp)
    return temp

def sanitize():
    time.sleep(2)
    GPIO.output(sanitizer,False)
    time.sleep(sanitize_delay)
    GPIO.output(sanitizer,True)

def start_conveyer():
    GPIO.output(conv_belt,False)
    time.sleep(2)
    GPIO.output(conv_belt,True)
    


def detect_and_predict_mask(frame, faceNet, maskNet):
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    #print(detections.shape)

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))


    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")
print("model")
i=1
try:
    while True:
        distance = check_dist()
        if distance <=8:
            temperature = check_temp()
            lcd.cursor_pos = (0,0)
            lcd.clear()
            time.sleep(0.5)
            lcd.write_string("Temperature: ")
            lcd.write_string(str(round(temperature)))
            time.sleep(1)
            if temperature>=100:
                lcd.cursor_pos = (0,0)
                lcd.clear()
                time.sleep(0.5)
                lcd.write_string("Temperature High!!!")
                time.sleep(2)
                lcd.cursor_pos = (0,0)
                lcd.clear()
                time.sleep(0.5)
                lcd.write_string("Sanitizing!!!")
                time.sleep(3)
                sanitize()
            
            elif temperature<100:
                lcd.cursor_pos = (0,0)
                lcd.clear()
                time.sleep(0.5)
                lcd.write_string("Sanitizing!!!")
                time.sleep(3)
                sanitize()
                lcd.clear()
                lcd.cursor_pos = (0,0)
                time.sleep(0.5)
                lcd.write_string("Please Face Towards Camera")
                time.sleep(1)
                
                camera = PiCamera()
                camera.resolution = (640, 480)
                rawCapture = PiRGBArray(camera,size=(640, 480))
                time.sleep(0.1)
                
                camera.capture(rawCapture, format="bgr")
                frame = rawCapture.array

                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
                for (box, pred) in zip(locs, preds):
                    
                    (mask, withoutMask) = pred
                    label = "Mask" if mask > withoutMask else "No Mask"
                    #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    pred = max(mask,withoutMask)*100
                    
                    print(pred,"  ",label)
                
                    
                    if label =="Mask":
                        GPIO.output(door_open,False)
                        GPIO.output(door_close,True)
                        lcd.cursor_pos = (0,0)
                        lcd.clear()
                        time.sleep(0.5)
                        lcd.write_string("Door Open")
                        time.sleep(2)
                        GPIO.output(door_open,True)
                        GPIO.output(door_close,False)
                        lcd.cursor_pos = (0,0)
                        lcd.clear()
                        time.sleep(0.5)
                        lcd.write_string("Door Close")
                        time.sleep(2)
                        GPIO.output(door_close,True)
                    
                    else:
                        lcd.cursor_pos = (0,0)
                        lcd.clear()
                        time.sleep(0.5)
                        lcd.write_string("Not allowedto enter w/o Mask")
                        time.sleep(2)
                        start_conveyer()
                        lcd.clear()
                        lcd.cursor_pos = (0,0)
                        time.sleep(0.5)
                        lcd.write_string("Take Your Mask from Conveyer Belt")
                        time.sleep(2)
                        lcd.cursor_pos = (0,0)
                        lcd.clear()
                        time.sleep(0.5)
                        lcd.write_string("Door Close")
                        time.sleep(2)
                cv2.imshow("frame",frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                camera.close()
                i=1
            
        else:
            if i==1:
                lcd.cursor_pos = (0,0)
                lcd.clear()
                time.sleep(0.5)
                lcd.write_string("Welcome!!!")
                lcd.cursor_pos = (1,0)
                lcd.write_string("Check Temp.")
                i=2
        time.sleep(1)

        
except KeyboardInterrupt:
    GPIO.cleanup()
    


