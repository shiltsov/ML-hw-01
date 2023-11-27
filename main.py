from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import HTTPException
from typing import Optional, List
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from fastapi.encoders import jsonable_encoder


import pandas as pd 
import pickle
import warnings

warnings.filterwarnings('ignore')

app = FastAPI()

scaler = pickle.load(open("./model/scaler.pkl", "rb"))
imp = pickle.load(open("./model/imp.pkl", "rb"))
ohe = pickle.load(open("./model/ohe.pkl", "rb"))
reg = pickle.load(open("./model/reg.pkl", "rb"))

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


# по идее тут нужно всю вообще предобработку сделать включая заполнение пропусков и тд
def make_prediction(items: List[Item]) :
    # создадим датафрейм
    df = pd.DataFrame(jsonable_encoder(items))
    df.drop(['name', 'torque','selling_price'], axis=1, inplace=True)

    # предобработка столбцов
    cols_to_split = ['mileage', 'engine', 'max_power']
    for col in cols_to_split:
        # меняем только те которые не пропущенные
        # в колонке 'max_power' попадается грязь - размерность без значения поэтому следим чтобы билось на 2
        ind = ~df[col].isna()
        df.loc[ind,col] = df.loc[ind,col].apply(lambda x: x.split()[0] if len(x.split()) == 2 else np.nan)

    # заполнение пропусков
    df[['mileage', 'engine', 'max_power','seats']] = imp.transform(df[['mileage', 'engine', 'max_power','seats']])    
    df['seats'] = df['seats'].astype('object')

    # шкалирование
    num_cols = df.select_dtypes(exclude=['object']).columns
    df[num_cols] =  scaler.transform(df[num_cols])

    # кодирование категориалных
    df_ohe = ohe.transform(df[["fuel","seller_type","transmission","owner","seats"]])
    df[ohe.get_feature_names_out()] = df_ohe
    df = df.drop(["fuel","seller_type","transmission","owner","seats"], axis=1)

    return reg.predict(df)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return make_prediction([item])[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return  make_prediction(items)

