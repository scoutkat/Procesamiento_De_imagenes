import cv2
import numpy as np
import pyautogui
import keyboard

fps = 10.0
resolucion = pyautogui.size()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('grabacion.mp4', fourcc, fps, resolucion)

while True:  # 'True' debe estar con la primera letra en mayúscula

    frame = np.array(pyautogui.screenshot())  
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)

    if keyboard.is_pressed('q'):
        break

out.release()
cv2.destroyAllWindows()
