import os
from dotenv import load_dotenv
import time


from src.scripts.pipe_counting import perform_infer
from src.scripts.pipe_counting import load_yolo_model
from fastapi import FastAPI, Request, Form, File, UploadFile
from typing import List, Optional
import cv2
import numpy as np
import io

load_dotenv()

image_path = os.getenv('IMAGE_PATH', "")
model_path = os.getenv('MODEL_PATH', "model.pt")
model = load_yolo_model(model_path=model_path)

app = FastAPI()




@app.post("/")
def detect_with_server_side_rendering(file: bytes = File(...)):
    stream = io.BytesIO(file)

    image = np.asarray(bytearray(stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite("test.jpg",image)
    image_path = "test.jpg"
    pipe_count = perform_infer(image_path=image_path, model=model)

    return pipe_count



if __name__ == '__main__':
    import uvicorn
    app_str = 'main:app'  # make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host='localhost', port=8000, reload=True)
