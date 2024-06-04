from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200"
]
app.add_middleware(

    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/1")
CLASS_NAMES = ["Coast","Desert","Forest","Glacier","Mountain"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

#calling this function in the post method below
def read_file_as_image(data) -> np.ndarray:
    #converting pillow image to numpy array
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
): 
    desired_size = (180, 180) 
    image = read_file_as_image(await file.read())
    image = cv2.resize(image,desired_size)
    img_batch = np.expand_dims(image,0) 
    
    #predict doesn't take single image as input. takes multiple so you need to convert it to batch
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(tf.nn.softmax(predictions   [0]))

    return {
        'class': predicted_class,
        'confidence':float(confidence)
    }

# if __name__ =="__main__":
    # uvicorn.run(app,host='localhost', port=8000)