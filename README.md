# Face-Mask-Detection-and-Temperature-Detection-with-Raspberry-Pi
WORKING:
All the senors and module required for this project have been names in the "circuit.jpg" file.
First of all if a Person comes in front of the "Ultrasonic Sensor" the Raspberry Pi recieves the information and prepares the "MLX Temperature Sensor" to sense the temperature of the person in front of ultrasonic sensor as the MLX sensor will be palced above the Ultrsonic Sensor then there arise two condition:


#First Condition:
  if the temperature is within the range of normal person's temperature the "Submersible 5volt motor will be turned on and sanitizer will come out of the tube after sanitization 
    Pi Camera will turn ON, take a snaphot of the person's face and with deep learnig model predits whether person is wearing a Mask or not. If Mask in ON then door will open for 
    some duration and then close and will be redy for new person. If Mask is ON the face of perosn then a conveyer belt motor will turn ON and the person will get a mask from the 
    conveyer belt and have to follow all the process again for entrance.

#Second Condition:
  if the temperature is greter then normal condition then sanitizer will will come out through pump motor but the person will not be allowed to enter inside the room

First of all dump OS into the Raspberry Pi and then import necesary packages into your Raspberry Pi: Python, Tensorflow, Numpy, Open CV, GPIO package
The actual working code lies inside the program.py file.
Folder named face_detector has two files named "deploy.prototxt" and "res10_300x300_ssd_iter_140000.caffemodel" that need to be loaded inide "program.py".
And a file name "mask_detector.model" also needs to be loaded by entering the path of file into the "program.py" file  which is a pre-trained model detecting whether the person is wearing mask or not.
Make connection according to the schematic given in "circuit.jpg" file.
After doing all these things run the "program.py" file from terminal and enjoy.

#################################Happpy Programming#############################################
