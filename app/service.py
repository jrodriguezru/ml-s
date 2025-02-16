from fastapi import FastAPI, HTTPException, UploadFile
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
import uvicorn

app = FastAPI()

# Load model
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {MODEL_PATH}!")
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}.")


# POST using a link to the CSV file as data
@app.post("/score1")
def score(data: str):
    try:
        input_data = pd.read_csv(data)
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# POST using a CSV file as data
@app.post("/score2")
def score(file: UploadFile):
    try:
        input_data = pd.read_csv(file.file)
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8888)
