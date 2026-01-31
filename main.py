from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet
from datetime import datetime

app = FastAPI(title="CryptoAnalyzer ML Engine")

class HistoricalPoint(BaseModel):
    date: str
    price: float
    sentiment: float

class ForecastRequest(BaseModel):
    coin_id: str
    history: List[HistoricalPoint]
    days_to_predict: int = 7

class ForecastPoint(BaseModel):
    date: str
    price: float

@app.post("/forecast", response_model=List[ForecastPoint])
async def generate_forecast(request: ForecastRequest):
    if len(request.history) < 30:
        raise HTTPException(status_code=400, detail="Мінімум 30 точок історії!")

    try:
        df = pd.DataFrame([{"ds": p.date, "y": p.price} for p in request.history])

        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None).dt.normalize()
        df = df.sort_values('ds')

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=request.days_to_predict)
        forecast = model.predict(future)

        predictions = forecast.tail(request.days_to_predict)
        
        result = []
        for _, row in predictions.iterrows():
            result.append(ForecastPoint(
                date=row['ds'].strftime('%Y-%m-%d'),
                price=round(float(row['yhat']), 2)
            ))

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)