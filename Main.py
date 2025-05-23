import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    #print(f'gesture recognition result: {result}')
    gesto = result.gestures[0][0]
    print(f'{timestamp_ms} Gesto: {gesto.category_name}')
    #print(f'Score: {result[0][0].score}')

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/Usuario/Desktop/Ejercicios/Imagen/Control-de-player-via-gestos/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with GestureRecognizer.create_from_options(options) as recognizer:
  cap = cv2.VideoCapture(1)
  print(cap)
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print('Error con la cámara')
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    recognizer.recognize_async(mp_image, timestamp_ms)

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()