import speech_recognition as sr

listener = sr.Recognizer()

try:
    with sr.Microphone(device_index=0) as source:
        voice = listener.listen(source)
        rec = listener.recognize_google(voice)
        print("Usted dijo: " , rec)
except:
    print("nothing")
