from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML-Engine")

app = FastAPI(title="CryptoAnalyzer ML Engine")

class HistoricalPoint(BaseModel):
    date: str
    price: float
    sentiment: float

class ForecastRequest(BaseModel):
    coin_id: str
    history: List[HistoricalPoint]
    days_to_predict: int

class ForecastPoint(BaseModel):
    date: str
    price: float

@app.post("/forecast", response_model=List[ForecastPoint])
async def generate_forecast(request: ForecastRequest):
    if len(request.history) < 30:
        raise HTTPException(status_code=400, detail="Мінімум 30 точок історії!")

    try:
        data = [{"ds": p.date, "y": p.price} for p in request.history]
        df = pd.DataFrame(data)

        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        df = df.sort_values('ds')

        requested_days = request.days_to_predict
        dynamic_window = max(30, requested_days * 5)
        
        df = df.tail(dynamic_window)
        
        logger.info(f"Generating forecast for {request.coin_id}. Points used: {len(df)}. Days to predict: {requested_days}")
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.21,
            changepoint_range=0.95 
        )

        model.fit(df)

        future = model.make_future_dataframe(periods=requested_days)
        forecast = model.predict(future)

        predictions = forecast.tail(requested_days)
        
        result = []
        for _, row in predictions.iterrows():
            result.append(ForecastPoint(
                date=row['ds'].strftime('%Y-%m-%d'),
                price=round(float(row['yhat']), 2)
            ))

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)