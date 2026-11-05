from datetime import date, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st 

BASE = Path(__file__).parent
OUT = BASE / "outputs"

st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚦", layout="wide")

# ------------------------------------------------------------ simple styling
st.markdown("""
<style>
#MainMenu, footer, .stDeployButton {visibility: hidden;}
.banner {padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;
         background: linear-gradient(90deg, #1f3c88, #3b82f6); color: white;}
.banner h1 {margin: 0; font-size: 32px; color: white;}
.banner p {margin: 6px 0 0 0; font-size: 16px; opacity: 0.9;}
.result {padding: 18px 20px; border-radius: 12px; background: rgba(128,128,128,0.12);}
</style>
""", unsafe_allow_html=True)

if not (BASE / "model.pkl").exists():
    st.error("model.pkl not found. Run  python train.py  first, then restart the app.")
    st.stop()


@st.cache_resource
def load_bundle():
    return joblib.load(BASE / "model.pkl")


b = load_bundle()
model, feature_cols, THR = b["model"], b["columns"], b["threshold"]
freq_maps, locs, defaults = b["freq_maps"], b["locations"], b["defaults"]
BASELINE = b["baseline"]

POI = {"Junction": "Junction", "Traffic_Signal": "Traffic signal", "Crossing": "Crossing",
       "Stop": "Stop sign", "Station": "Station", "Amenity": "Amenity", "Railway": "Railway",
       "Give_Way": "Give way", "Bump": "Speed bump", "Roundabout": "Roundabout",
       "No_Exit": "No exit", "Traffic_Calming": "Traffic calming"}
WEATHER = ["Clear", "Cloudy", "Rain", "Snow/Ice", "Fog/Haze", "Other"]

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def box():
    """A bordered box (falls back to a plain container on older Streamlit)."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def build_features(inp):
    """Same steps as train.py: derived time features, frequency encoding, one-hot encoding."""
    row = dict(inp)
    row["IsWeekend"] = int(row["DayOfWeek"] >= 5)
    row["IsRushHour"] = int(row["Hour"] in [7, 8, 9, 16, 17, 18] and row["IsWeekend"] == 0)
    for c in ["City", "County", "State"]:
        row[c + "_freq"] = freq_maps[c].get(str(row[c]), 0.0)
    one = pd.DataFrame([row]).drop(columns=["City", "County", "State"])
    one = pd.get_dummies(one, columns=["Weather_Group", "Sunrise_Sunset"], dtype=np.int8)
    return one.reindex(columns=feature_cols, fill_value=0).astype(np.float32)


def predict_many(inputs):
    X = pd.concat([build_features(i) for i in inputs], ignore_index=True)
    return model.predict_proba(X)[:, 1]


def risk_level(prob):
    """HIGH if probability >= threshold, MODERATE if it is close, otherwise LOW."""
    if prob >= THR:
        return "HIGH", "#e74c3c", "Long traffic delay expected (severity 3-4)"
    if prob >= THR * 0.7:
        return "MODERATE", "#f39c12", "Close to the high-impact threshold"
    return "LOW", "#2ecc71", "Short traffic delay expected (severity 1-2)"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🚦 Accident Severity")
    st.caption("Predicts whether an accident will cause a HIGH or LOW traffic impact.")
    st.markdown("**How to use**")
    st.markdown("1. Choose the location\n2. Set the time and weather\n"
                "3. Pick nearby road features\n4. Click **Predict severity**")
    st.divider()
    st.markdown(f"**Model:** {b['model_name']}")
    st.markdown(f"**Threshold:** {THR:.2f}")
    st.markdown(f"**Baseline accuracy:** {BASELINE * 100:.1f}%")
    st.divider()
    st.caption("Mini project · US Accidents dataset (2016-2023)")

st.markdown("""
<div class="banner">
  <h1>🚦 Traffic Accident Severity Predictor</h1>
  <p>Enter the conditions of an accident and the model predicts its traffic impact.</p>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_ins, tab_about = st.tabs(["🔮 Predict", "📊 Model Insights", "ℹ️ About"])

# ------------------------------------------------------------- predict tab
with tab_pred:
    c1, c2, c3 = st.columns(3)

    with c1:
        with box():
            st.subheader("📍 Location")
            state = st.selectbox("State",
                                 sorted(locs["State"].unique(), key=lambda s: STATE_NAMES.get(s, s)),
                                 format_func=lambda s: STATE_NAMES.get(s, s))
            s_locs = locs[locs["State"] == state]
            city = st.selectbox("City", sorted(s_locs["City"].unique()))
            loc = s_locs[s_locs["City"] == city].iloc[0]
            st.map(pd.DataFrame({"lat": [loc["Start_Lat"]], "lon": [loc["Start_Lng"]]}), height=220)

    with c2:
        with box():
            st.subheader("🕒 Time")
            d = st.date_input("Date", date(2022, 6, 15))
            t = st.time_input("Time", time(17, 30))
            light = st.radio("Light condition", ["Day", "Night"], horizontal=True)
        with box():
            st.subheader("🛣️ Road features nearby")
            chosen = st.multiselect("Select all that apply", list(POI.values()))

    with c3:
        with box():
            st.subheader("🌦️ Weather")
            weather = st.selectbox("Weather", WEATHER)
            temp = st.slider("Temperature (°F)", -20, 115, int(clip(defaults["Temperature(F)"], -20, 115)))
            hum = st.slider("Humidity (%)", 0, 100, int(clip(defaults["Humidity(%)"], 0, 100)))
            vis = st.slider("Visibility (miles)", 0.0, 10.0,
                            float(clip(defaults["Visibility(mi)"], 0, 10)), 0.5)
            wind = st.slider("Wind speed (mph)", 0, 60, int(clip(defaults["Wind_Speed(mph)"], 0, 60)))
            p1, p2 = st.columns(2)
            pres = p1.number_input("Pressure (in)", 25.0, 32.0,
                                   float(clip(defaults["Pressure(in)"], 25, 32)), 0.1)
            precip = p2.number_input("Rain (in)", 0.0, 3.0, 0.0, 0.01)

    inp = {"Start_Lat": float(loc["Start_Lat"]), "Start_Lng": float(loc["Start_Lng"]),
           "City": loc["City"], "County": loc["County"], "State": loc["State"],
           "Temperature(F)": temp, "Humidity(%)": hum, "Pressure(in)": pres,
           "Visibility(mi)": vis, "Wind_Speed(mph)": wind, "Precipitation(in)": precip,
           "Weather_Group": weather, "Sunrise_Sunset": light,
           "Hour": t.hour, "DayOfWeek": d.weekday(), "Month": d.month}
    for key, label in POI.items():
        inp[key] = int(label in chosen)

    st.write("")
    go = st.button("🔍 Predict severity", type="primary", use_container_width=True)

    if go:
        prob = float(predict_many([inp])[0])
        level, color, message = risk_level(prob)

        st.markdown("### Result")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"""
<div class="result" style="border-left: 8px solid {color};">
  <div style="font-size:14px; opacity:0.7;">Predicted traffic impact</div>
  <div style="font-size:36px; font-weight:700; color:{color};">{level}</div>
  <div style="font-size:15px;">{message}</div>
</div>
""", unsafe_allow_html=True)
            st.write("")
            st.metric("Probability of high impact", f"{prob * 100:.1f}%")
            st.progress(min(prob, 1.0))
            st.caption(f"Marked HIGH when the probability is at least {THR * 100:.0f}%.")
            st.caption(f"{STATE_NAMES.get(state, state)}, {city} · {t.strftime('%H:%M')} · "
                       f"{weather} · {len(chosen)} road feature(s)")

        with r2:
            st.markdown("**What-if: risk at each hour of the day** (other inputs unchanged)")
            hr = pd.DataFrame({"Risk": predict_many([dict(inp, Hour=h) for h in range(24)]),
                               "Threshold": THR},
                              index=pd.Index(range(24), name="Hour"))
            st.line_chart(hr)

        st.markdown("**What-if: risk under each weather condition**")
        wx = pd.DataFrame({"Risk": predict_many([dict(inp, Weather_Group=w) for w in WEATHER])},
                          index=WEATHER)
        st.bar_chart(wx)
    else:
        st.info("Choose the conditions above and click **Predict severity**.")

# ------------------------------------------------------------ insights tab
with tab_ins:
    st.header("Model performance")
    csv = OUT / "model_comparison.csv"
    acc_csv = OUT / "train_test_accuracy.csv"

    if csv.exists():
        res = pd.read_csv(csv)
        final = res[res["Model"] == "Random Forest"].iloc[0]

        # --- train vs test accuracy
        st.subheader("Train vs test accuracy (Random Forest)")
        if acc_csv.exists():
            acc = pd.read_csv(acc_csv)
            rf_acc = acc[acc["Model"] == "Random Forest"].iloc[0]
            gap = (rf_acc["Train_Accuracy"] - rf_acc["Test_Accuracy"]) * 100
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Train accuracy", f"{rf_acc['Train_Accuracy'] * 100:.1f}%")
            a2.metric("Test accuracy", f"{rf_acc['Test_Accuracy'] * 100:.1f}%")
            a3.metric("Gap (train - test)", f"{gap:.1f} pts")
            a4.metric("Baseline accuracy", f"{BASELINE * 100:.1f}%")
            st.dataframe(acc, hide_index=True, use_container_width=True)
            if gap > 10:
                st.warning("Train accuracy is much higher than test accuracy: the model may be overfitting.")
            else:
                st.caption("A small gap between train and test accuracy means the model is not overfitting badly.")
        else:
            st.info("Run  python train.py  to create outputs/train_test_accuracy.csv")

        # --- test metrics
        st.subheader("Test set metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{final['Accuracy'] * 100:.1f}%")
        m2.metric("Macro F1", f"{final['Macro_F1']:.3f}")
        m3.metric("Recall (high)", f"{final['Recall']:.3f}")
        m4.metric("ROC-AUC", f"{final['ROC_AUC']:.3f}")
        st.dataframe(res, hide_index=True, use_container_width=True)
        st.caption("Accuracy alone is misleading on imbalanced data (about 80% of accidents "
                   "are low impact), so macro F1, recall and ROC-AUC are shown too.")
    else:
        st.info("Run train.py to create outputs/model_comparison.csv")

    for title, fname in [("Exploratory analysis", "eda.png"),
                         ("Confusion matrix (from test.py)", "test_confusion_matrix.png"),
                         ("Most important features", "feature_importance.png")]:
        if (OUT / fname).exists():
            with box():
                st.subheader(title)
                st.image(str(OUT / fname))

# about tab
with tab_about:
    st.header("About this project")
    st.markdown("""
**Problem:** Predict whether a road accident will have a high traffic impact (severity 3-4)
or a low impact (severity 1-2), using only information known when the accident happens.

**Dataset:** US Accidents (2016-2023) by Sobhan Moosavi (Kaggle). A random sample was used.

**Method:** cleaning → time, weather and location features → Logistic Regression (baseline)
and Random Forest (final model) → evaluation on a held-out test set.

**Limitations:**
- Severity in this dataset measures traffic delay, not injuries.
- Only reported accidents are included.
- The model shows association, not causation.
- A US-trained model may not transfer directly to Indian roads.

**Future work:** Indian accident data, live weather, hotspot mapping and time-based forecasting.
""")