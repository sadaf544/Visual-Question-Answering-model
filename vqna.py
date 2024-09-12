import cv2

import requests

from PIL import Image

import time

from transformers import BlipProcessor, BlipForQuestionAnswering

import speech_recognition as sr

import torch

from gtts import gTTS

import pyttsx3

 

# Load the processor and model

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

 

# Initialize the webcam

cap = cv2.VideoCapture(0)  # 0 is the default camera

 

# Check if the webcam is opened correctly

if not cap.isOpened():

    print("Error: Could not open webcam.")

    exit()

 

# Initialize text-to-speech engine

engine = pyttsx3.init()

voices = engine.getProperty('voices')

engine.setProperty('voice', voices[0].id)

 

def listen_for_command():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("Speak your command:")

        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source)

 

    try:

        command = recognizer.recognize_google(audio)

        print("You said: " + command)

        return command

    except sr.UnknownValueError:

        print("Could not understand audio")

        return None

    except sr.RequestError as e:

        print("Could not request results from Google Speech Recognition service; {0}".format(e))

        return None

 

question = None

 

while True:

    # Capture frame-by-frame

    ret, frame = cap.read()

 

    if not ret:

        print("Failed to grab frame")

        break

 

    # Display the frame

    cv2.imshow('Webcam', frame)

 

    # Listen for stop command or new question

    command = listen_for_command()

 

    if command and "stop" in command.lower():

        print("Stopping the program...")

        break

    elif command:

        question = command

 

    # If a question is provided, process the frame

    if question:

        # Convert the frame to PIL image

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

 

        # Process inputs

        inputs = processor(pil_image, question, return_tensors="pt")

 

        # Make predictions using the generate method

        with torch.no_grad():

            generated_ids = model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)

 

        # Get the predicted answer

        predicted_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(predicted_answer)

 

        # Speak the predicted answer

        engine.say(predicted_answer)

        engine.runAndWait()

 

        # Reset the question after answering

        question = None

 

    # Press 'q' on the keyboard to exit

    if cv2.waitKey(1) & 0xFF == ord('q'):

        print("Quitting...")

        break

 

# Release the webcam and close all OpenCV windows

cap.release()

cv2.destroyAllWindows()
