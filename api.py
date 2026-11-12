# api.py : a REST API that serves the saved model.

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).parent

# ---- load the model saved by train.py
bundle = joblib.load(BASE / "model.pkl")
model, feature_cols, THR = bundle["model"], bundle["columns"], bundle["threshold"]
freq_maps, locs = bundle["freq_maps"], bundle["locations"]

ROAD_FEATURES = ["Junction", "Traffic_Signal", "Crossing", "Stop", "Station", "Amenity",
                 "Railway", "Give_Way", "Bump", "Roundabout", "No_Exit", "Traffic_Calming"]
WEATHER = ["Clear", "Cloudy", "Rain", "Snow/Ice", "Fog/Haze", "Other"]

app = FastAPI(title="Accident Severity API",
              description="Predicts whether an accident will have a HIGH or LOW traffic impact.")


# ---- the input format (FastAPI checks it automatically)
class AccidentInput(BaseModel):
    state: str = "CA"                         # state code, e.g. CA
    city: str = "Los Angeles"
    hour: int = Field(17, ge=0, le=23)
    day_of_week: int = Field(2, ge=0, le=6)   # 0 = Monday ... 6 = Sunday
    month: int = Field(6, ge=1, le=12)
    light: str = "Day"                        # Day or Night
    weather: str = "Clear"                    # Clear, Cloudy, Rain, Snow/Ice, Fog/Haze, Other
    temperature: float = 70                   # Fahrenheit
    humidity: float = 50
    pressure: float = 29.9
    visibility: float = 10
    wind_speed: float = 5
    precipitation: float = 0
    road_features: List[str] = []             # e.g. ["Junction", "Traffic_Signal"]


def build_features(inp: dict) -> pd.DataFrame:
    """Same steps as train.py: time features, frequency encoding, one-hot encoding."""
    row = dict(inp)
    row["IsWeekend"] = int(row["DayOfWeek"] >= 5)
    row["IsRushHour"] = int(row["Hour"] in [7, 8, 9, 16, 17, 18] and row["IsWeekend"] == 0)
    for c in ["City", "County", "State"]:
        row[c + "_freq"] = freq_maps[c].get(str(row[c]), 0.0)
    one = pd.DataFrame([row]).drop(columns=["City", "County", "State"])
    one = pd.get_dummies(one, columns=["Weather_Group", "Sunrise_Sunset"], dtype=np.int8)
    return one.reindex(columns=feature_cols, fill_value=0).astype(np.float32)


@app.get("/")
def home():
    return {"message": "Accident Severity API is running. Open /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["model_name"], "threshold": THR}


@app.get("/cities/{state}")
def cities(state: str):
    """List the cities available for a state code."""
    found = sorted(locs[locs["State"] == state.upper()]["City"].unique().tolist())
    if not found:
        raise HTTPException(status_code=404, detail="State code not found")
    return {"state": state.upper(), "cities": found}


@app.post("/predict")
def predict(data: AccidentInput):
    if data.weather not in WEATHER:
        raise HTTPException(status_code=422, detail=f"weather must be one of {WEATHER}")
    if data.light not in ["Day", "Night"]:
        raise HTTPException(status_code=422, detail="light must be Day or Night")
    bad = [r for r in data.road_features if r not in ROAD_FEATURES]
    if bad:
        raise HTTPException(status_code=422, detail=f"unknown road features {bad}. Use {ROAD_FEATURES}")

    # find the city's coordinates in the saved table
    match = locs[(locs["State"] == data.state.upper()) &
                 (locs["City"].str.lower() == data.city.lower())]
    if match.empty:
        raise HTTPException(status_code=404,
                            detail="City not found. Use /cities/{state} to see valid cities.")
    loc = match.iloc[0]

    inp = {"Start_Lat": float(loc["Start_Lat"]), "Start_Lng": float(loc["Start_Lng"]),
           "City": loc["City"], "County": loc["County"], "State": loc["State"],
           "Temperature(F)": data.temperature, "Humidity(%)": data.humidity,
           "Pressure(in)": data.pressure, "Visibility(mi)": data.visibility,
           "Wind_Speed(mph)": data.wind_speed, "Precipitation(in)": data.precipitation,
           "Weather_Group": data.weather, "Sunrise_Sunset": data.light,
           "Hour": data.hour, "DayOfWeek": data.day_of_week, "Month": data.month}
    for r in ROAD_FEATURES:
        inp[r] = int(r in data.road_features)

    prob = float(model.predict_proba(build_features(inp))[:, 1][0])
    if prob >= THR:
        level = "HIGH"
    elif prob >= THR * 0.7:
        level = "MODERATE"
    else:
        level = "LOW"

    return {"probability_high_impact": round(prob, 4),
            "threshold": THR,
            "predicted_class": int(prob >= THR),     # 1 = high impact, 0 = low impact
            "risk_level": level}