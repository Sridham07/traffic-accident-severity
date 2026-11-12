from datetime import date, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
OUT = BASE / "outputs"

st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
#MainMenu, footer, .stDeployButton {visibility: hidden;}
.banner {padding: 26px 30px; border-radius: 16px; margin-bottom: 18px;
         background: linear-gradient(90deg, #1f3c88, #3b82f6); color: white;}
.banner h1 {margin: 0; font-size: 34px; color: white;}
.banner p {margin: 8px 0 0 0; font-size: 17px; opacity: 0.92;}
.step {font-weight: 700; font-size: 18px; margin-bottom: 2px;}
.hint {opacity: 0.7; font-size: 14px; margin-bottom: 10px;}
.result {padding: 20px 22px; border-radius: 14px; background: rgba(128,128,128,0.12);}
.big {font-size: 40px; font-weight: 800; line-height: 1.1;}
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
BASELINE = b["baseline"]                    # share of LOW-impact accidents in the test data
AVG_HIGH = 1 - BASELINE                     # share of HIGH-impact accidents (the "average" chance)

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


# ------------------------------------------------------------------ helpers
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
    """Returns level, color, headline and a short plain-language note."""
    if prob >= THR:
        return ("HIGH", "#e74c3c", "This looks like a high-impact accident.",
                "The model expects a long traffic delay in these conditions.")
    if prob >= THR * 0.7:
        return ("MODERATE", "#f39c12", "Not high, but close to the line.",
                "The delay could go either way, so conditions like these deserve some caution.")
    return ("LOW", "#2ecc71", "This looks like a low-impact accident.",
            "The model expects traffic to recover fairly quickly.")


def gauge(prob, thr, color):
    """A simple risk bar with the cut-off marked."""
    return f"""
<div style="position:relative; height:18px; border-radius:9px; background:rgba(128,128,128,0.25);
            margin:14px 0 34px 0;">
  <div style="width:{min(prob, 1) * 100:.1f}%; height:100%; border-radius:9px; background:{color};"></div>
  <div style="position:absolute; left:{thr * 100:.1f}%; top:-5px; height:28px; border-left:3px solid #7f8c8d;"></div>
  <div style="position:absolute; left:{thr * 100:.1f}%; top:26px; font-size:12px; opacity:0.8;
              transform:translateX(-50%);">cut-off {thr * 100:.0f}%</div>
</div>"""


def compare_line(label, other, prob):
    diff = (prob - other) * 100
    if abs(diff) < 0.5:
        return f"{label}: about the same ({other * 100:.1f}%)."
    word = "higher" if diff > 0 else "lower"
    return f"{label}: {other * 100:.1f}%, so yours is {abs(diff):.1f} points {word}."


# ---------------------------------------- form defaults and quick examples
DEFAULTS = {
    "day": date(2022, 6, 15), "tm": time(17, 30), "light": "Day", "roads": [], "weather": "Clear",
    "temp": int(clip(defaults["Temperature(F)"], -20, 115)),
    "hum": int(clip(defaults["Humidity(%)"], 0, 100)),
    "vis": float(round(clip(defaults["Visibility(mi)"], 0, 10) * 2) / 2),
    "wind": int(clip(defaults["Wind_Speed(mph)"], 0, 60)),
    "pres": float(round(clip(defaults["Pressure(in)"], 25, 32), 1)),
    "precip": 0.0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("history", [])

PRESETS = {
    "Rainy rush hour": {"tm": time(17, 30), "light": "Day", "weather": "Rain", "hum": 90,
                        "vis": 4.0, "precip": 0.2,
                        "roads": ["Junction", "Traffic signal", "Crossing"]},
    "Snowy morning": {"tm": time(8, 0), "light": "Day", "weather": "Snow/Ice", "temp": 25,
                      "hum": 85, "vis": 2.0, "precip": 0.1, "roads": ["Junction"]},
    "Quiet clear night": {"tm": time(2, 0), "light": "Night", "weather": "Clear", "roads": []},
    "Sunny afternoon": {"tm": time(14, 0), "light": "Day", "weather": "Clear",
                        "roads": ["Traffic signal"]},
}


def apply_preset(name):
    st.session_state.update({**DEFAULTS, **PRESETS[name]})


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("Accident Severity")
    st.caption("A machine learning demo that estimates how badly an accident will disrupt traffic.")
    st.markdown("**How to use it**")
    st.markdown("1. Pick an example or fill in the 4 steps\n"
                "2. Click **Predict**\n"
                "3. Read the result and the 'why' section")
    st.divider()
    st.markdown(f"**Model:** {b['model_name']}")
    st.markdown(f"**Cut-off for HIGH:** {THR * 100:.0f}% chance")
    st.divider()
    st.caption("Mini project | US Accidents dataset (2016-2023)")

st.markdown("""
<div class="banner">
  <h1>Traffic Accident Severity Predictor</h1>
  <p>Tell us where, when and in what conditions an accident happens, and we'll estimate
     how long it is likely to disrupt traffic.</p>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_ins, tab_about = st.tabs(["Predict", "How good is the model?", "About"])

# ------------------------------------------------------------- predict tab
with tab_pred:
    st.markdown("#### In a hurry? Try a quick example")
    p_cols = st.columns(len(PRESETS))
    for col, name in zip(p_cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        with box():
            st.markdown('<div class="step">Step 1: Where?</div>'
                        '<div class="hint">Choose the state and city.</div>', unsafe_allow_html=True)
            state = st.selectbox("State",
                                 sorted(locs["State"].unique(), key=lambda s: STATE_NAMES.get(s, s)),
                                 format_func=lambda s: STATE_NAMES.get(s, s))
            s_locs = locs[locs["State"] == state]
            city = st.selectbox("City", sorted(s_locs["City"].unique()))
            loc = s_locs[s_locs["City"] == city].iloc[0]
            st.map(pd.DataFrame({"lat": [loc["Start_Lat"]], "lon": [loc["Start_Lng"]]}), height=200)

    with c2:
        with box():
            st.markdown('<div class="step">Step 2: When?</div>'
                        '<div class="hint">Rush hours and darkness change traffic a lot.</div>',
                        unsafe_allow_html=True)
            d = st.date_input("Date", key="day")
            t = st.time_input("Time", key="tm")
            light = st.radio("Light condition", ["Day", "Night"], horizontal=True, key="light")
        with box():
            st.markdown('<div class="step">Step 3: What is nearby?</div>'
                        '<div class="hint">Pick everything close to the accident spot.</div>',
                        unsafe_allow_html=True)
            chosen = st.multiselect("Road features", list(POI.values()), key="roads")

    with c3:
        with box():
            st.markdown('<div class="step">Step 4: What is the weather like?</div>'
                        '<div class="hint">Leave the sliders alone if you are not sure.</div>',
                        unsafe_allow_html=True)
            weather = st.selectbox("Weather", WEATHER, key="weather")
            temp = st.slider("Temperature (F)", -20, 115, key="temp")
            hum = st.slider("Humidity (%)", 0, 100, key="hum")
            vis = st.slider("Visibility (miles)", 0.0, 10.0, step=0.5, key="vis")
            wind = st.slider("Wind speed (mph)", 0, 60, key="wind")
            q1, q2 = st.columns(2)
            pres = q1.number_input("Pressure (in)", 25.0, 32.0, step=0.1, key="pres")
            precip = q2.number_input("Rain (in)", 0.0, 3.0, step=0.01, key="precip")

    inp = {"Start_Lat": float(loc["Start_Lat"]), "Start_Lng": float(loc["Start_Lng"]),
           "City": loc["City"], "County": loc["County"], "State": loc["State"],
           "Temperature(F)": temp, "Humidity(%)": hum, "Pressure(in)": pres,
           "Visibility(mi)": vis, "Wind_Speed(mph)": wind, "Precipitation(in)": precip,
           "Weather_Group": weather, "Sunrise_Sunset": light,
           "Hour": t.hour, "DayOfWeek": d.weekday(), "Month": d.month}
    for key, label in POI.items():
        inp[key] = int(label in chosen)

    st.write("")
    go = st.button("Predict what will happen", type="primary", use_container_width=True)

    if go:
        # one batch call: your input + three "what if" cases + 24 hours + 6 weather types
        batch = [inp,
                 dict(inp, Weather_Group="Clear"),
                 dict(inp, **{k: 0 for k in POI}),
                 dict(inp, Hour=3)]
        batch += [dict(inp, Hour=h) for h in range(24)]
        batch += [dict(inp, Weather_Group=w) for w in WEATHER]
        p = predict_many(batch)

        place = f"{city}, {STATE_NAMES.get(state, state)}"
        when = f"{d.strftime('%A, %d %B %Y')} at {t.strftime('%I:%M %p')}"
        roads_txt = ", ".join(chosen) if chosen else "none selected"
        st.session_state["last"] = {
            "prob": float(p[0]), "p_clear": float(p[1]), "p_noroad": float(p[2]),
            "p_3am": float(p[3]), "hours": p[4:28], "wx": p[28:],
            "place": place, "when": when, "weather": weather, "roads": chosen,
            "roads_txt": roads_txt, "hour": t.hour}
        st.session_state["history"].append(
            {"Place": place, "Time": t.strftime("%I:%M %p"), "Weather": weather,
             "Road features": roads_txt, "Chance of high impact": f"{p[0] * 100:.1f}%",
             "Result": risk_level(float(p[0]))[0]})

    last = st.session_state.get("last")
    if last:
        prob = last["prob"]
        level, color, headline, note = risk_level(prob)

        st.divider()
        st.markdown(f"### Result for {last['place']}")
        st.caption(f"{last['when']} | {last['weather']} weather | road features: {last['roads_txt']}")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
<div class="result" style="border-left: 8px solid {color};">
  <div style="font-size:14px; opacity:0.7;">Predicted traffic impact</div>
  <div class="big" style="color:{color};">{level}</div>
  <div style="font-size:17px; margin-top:6px;"><b>{headline}</b></div>
  <div style="font-size:15px; opacity:0.85; margin-top:4px;">{note}</div>
</div>
""", unsafe_allow_html=True)
            st.markdown(gauge(prob, THR, color), unsafe_allow_html=True)
            st.markdown(f"**{prob * 100:.1f}%** chance of a high-impact accident. "
                        f"An average accident in the data has about **{AVG_HIGH * 100:.1f}%**.")

        with r2:
            st.markdown("#### Why? What if things were different")
            if last["weather"] != "Clear":
                st.markdown("- " + compare_line("If the weather were Clear", last["p_clear"], prob))
            if last["roads"]:
                st.markdown("- " + compare_line("If there were no road features nearby",
                                                last["p_noroad"], prob))
            if last["hour"] != 3:
                st.markdown("- " + compare_line("If it happened at 3 AM", last["p_3am"], prob))
            st.caption("These are the model's estimates. They show patterns in past data, "
                       "not proven causes.")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**How the chance changes through the day**")
            st.line_chart(pd.DataFrame({"Chance of high impact": last["hours"], "Cut-off": THR},
                                       index=pd.Index(range(24), name="Hour of day")))
        with g2:
            st.markdown("**How the chance changes with the weather**")
            st.bar_chart(pd.DataFrame({"Chance of high impact": last["wx"]}, index=WEATHER))

        summary = (f"Traffic accident severity prediction\n"
                   f"Place: {last['place']}\nWhen: {last['when']}\nWeather: {last['weather']}\n"
                   f"Road features: {last['roads_txt']}\n"
                   f"Result: {level} ({prob * 100:.1f}% chance of high impact, "
                   f"cut-off {THR * 100:.0f}%)\n")
        st.download_button("Download this result", summary, file_name="prediction.txt")

    if st.session_state["history"]:
        with st.expander(f"Your recent predictions ({len(st.session_state['history'])})"):
            st.dataframe(pd.DataFrame(st.session_state["history"][-10:][::-1]),
                         hide_index=True, use_container_width=True)

# ------------------------------------------------------------ insights tab
with tab_ins:
    st.header("How good is the model?")
    st.caption("All scores below come from accidents the model never saw during training.")
    csv = OUT / "model_comparison.csv"
    acc_csv = OUT / "train_test_accuracy.csv"

    if csv.exists():
        res = pd.read_csv(csv)
        final = res[res["Model"] == "Random Forest"].iloc[0]

        st.subheader("Scores for the final model (Random Forest)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{final['Accuracy'] * 100:.1f}%")
        m2.metric("Macro F1", f"{final['Macro_F1']:.3f}")
        m3.metric("Recall (high)", f"{final['Recall']:.3f}")
        m4.metric("ROC-AUC", f"{final['ROC_AUC']:.3f}")
        st.caption(f"For comparison, always guessing 'low impact' would score "
                   f"about {BASELINE * 100:.1f}% accuracy.")

        with st.expander("What do these numbers mean?"):
            st.markdown("""
- **Accuracy:** out of 100 accidents, how many were predicted correctly.
- **Recall (high):** out of all real high-impact accidents, how many the model found.
- **Macro F1:** one balanced score that treats high and low impact accidents equally.
- **ROC-AUC:** how well the model ranks risky accidents above safe ones (0.5 = guessing, 1.0 = perfect).

About 8 in 10 accidents are low impact, so accuracy alone can look good even for a weak model.
That is why recall, macro F1 and ROC-AUC are shown too.
""")

        st.subheader("Is the model overfitting?")
        if acc_csv.exists():
            acc = pd.read_csv(acc_csv)
            rf_acc = acc[acc["Model"] == "Random Forest"].iloc[0]
            gap = (rf_acc["Train_Accuracy"] - rf_acc["Test_Accuracy"]) * 100
            a1, a2, a3 = st.columns(3)
            a1.metric("Train accuracy", f"{rf_acc['Train_Accuracy'] * 100:.1f}%")
            a2.metric("Test accuracy", f"{rf_acc['Test_Accuracy'] * 100:.1f}%")
            a3.metric("Gap", f"{gap:.1f} pts")
            if gap > 10:
                st.warning("Train accuracy is much higher than test accuracy, "
                           "so the model may be memorising the training data.")
            else:
                st.caption("A small gap means the model works about as well on new accidents "
                           "as on the ones it learned from.")
        else:
            st.info("Run  python train.py  to create outputs/train_test_accuracy.csv")

        st.subheader("Both models side by side")
        st.dataframe(res, hide_index=True, use_container_width=True)
    else:
        st.info("Run train.py to create outputs/model_comparison.csv")

    for title, fname in [("What the data looks like", "eda.png"),
                         ("Where the model gets it right and wrong", "test_confusion_matrix.png"),
                         ("What the model pays most attention to", "feature_importance.png")]:
        if (OUT / fname).exists():
            with box():
                st.subheader(title)
                st.image(str(OUT / fname))

# ---------------------------------------------------------------- about tab
with tab_about:
    st.header("About this project")
    st.markdown("""
**The question:** when a road accident happens, will it cause a long traffic delay (high impact)
or a short one (low impact)?

**The data:** US Accidents (2016-2023) by Sobhan Moosavi (Kaggle). A random sample was used.

**How it works:** the data is cleaned, time, weather and location features are created, and two
models are trained: Logistic Regression (a simple baseline) and Random Forest (the final model).
They are tested on accidents they have never seen.

**Good to know**
- "Severity" in this dataset means **traffic delay**, not how badly people were hurt.
- Only reported accidents are included.
- The model finds patterns, not causes.
- It was trained on US data, so it may not fit Indian roads.

**Future work:** Indian accident data, live weather, hotspot maps and time-based forecasting.
""")