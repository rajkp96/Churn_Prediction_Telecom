import pandas as pd
import joblib

model=joblib.load("model/churn_model.pkl")
features=joblib.load("model/features.pkl")

def predict_churn(data:dict):
    df=pd.dataframe([Data])
    df=pd.get_dummies(df)
    df=df.reindex(columns=features,fill_value=0)

    prediction = model.predict(df)[o]
    return "Yes" if prediction ==1 else "No"