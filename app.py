# streamlit run oil_predict.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title(" PH Fuel Price Tracker")

data = [
    {"date": "2026-03-20", "gasoline": 75, "diesel": 85},
    {"date": "2026-03-23", "gasoline": 85, "diesel": 100},
    {"date": "2026-03-26", "gasoline": 91.6, "diesel": 114.9},
]

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

df['days'] = (df['date'] - df['date'].min()).dt.days
X = df[['days']]

# Predict 7 days ahead
predictions = {}
future_day = pd.DataFrame({'days': [df['days'].max() + 7]})

for fuel in ['gasoline', 'diesel']:
    y = df[fuel]
    model = LinearRegression()
    model.fit(X, y)
    predictions[fuel] = model.predict(future_day)[0]

future_date = df['date'].max() + pd.Timedelta(days=7)
pred_df = pd.DataFrame({
    "date": [future_date],
    "gasoline": [predictions['gasoline']],
    "diesel": [predictions['diesel']]
})

# --- Display historical chart ---
st.subheader("Historical Fuel Prices")
st.line_chart(df.set_index('date')[['gasoline','diesel']])

# --- Display tables ---
st.subheader("Latest Prices")
st.table(df.tail(1)[['date','gasoline','diesel']])

st.subheader("Next Prices(Predicted)")
st.table(pred_df)