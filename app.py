from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from model import predict_price, get_feature_importance

app = FastAPI(title="AirBnB Price Predictor API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    city: str
    accommodates: int
    bedrooms: int
    bathrooms: float
    beds: int
    room_type: str
    property_type: str
    cancellation_policy: str
    cleaning_fee: float
    review_scores_rating: Optional[float] = 90.0
    number_of_reviews: Optional[int] = 10
    host_response_rate: Optional[float] = 95.0
    host_identity_verified: Optional[bool] = True


class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: dict
    top_features: list
    city_comparison: dict


@app.get("/")
def root():
    return {"message": "AirBnB Price Prediction API", "status": "running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = predict_price(request.dict())

        return PredictionResponse(
            predicted_price=result['price'],
            confidence_interval=result['confidence_interval'],
            top_features=result['top_features'],
            city_comparison=result['city_comparison']
        )
    except Exception as e:
        # Print full traceback
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/cities")
def get_cities():
    return {"cities": ["LA", "SF", "NYC", "Chicago", "Boston", "DC"]}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)