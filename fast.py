from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import pandas as pd
from recommender import recommend_programs

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r") as file:
        return file.read()

# Load the model and data
with open('recommender_model.pkl', 'rb') as file:
    vectorizer, program_tfidf, master_program_df = pickle.load(file)

# Define the input model
class CandidateInput(BaseModel):
    current_job: str
    career_interest: str
    qualification: str

# Endpoint to get recommendations
@app.post("/recommend")
def get_recommendations(input_data: CandidateInput):
    try:
        recommendations = recommend_programs(
            input_data.current_job,
            input_data.career_interest,
            input_data.qualification,
            vectorizer,
            program_tfidf,
            master_program_df
        )
        return recommendations.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
