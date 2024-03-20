from typing import Union
from fastapi import FastAPI
import pandas as pd
import numpy as np
import json
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

## Datasets for API functions

df_function_PlayTimeGenre=pd.read_parquet("Dataset/Function_Dataset/F1.parquet") ## Function 1
df_function_UserForGenre=pd.read_parquet("Dataset/Function_Dataset/F2.parquet")## Function 2
df_function_Recommends=pd.read_parquet("Dataset/Function_Dataset/F3-4-5.parquet")## Function 3 and 4
df_function_Sentiment=pd.read_parquet("Dataset/Function_Dataset/F5.parquet")## Function 5
df_function_Item_Item=pd.read_parquet("Dataset/Function_Dataset/MLF1.parquet")## Function ML
##Functions development in https://github.com/Abyzou1995/PI1_ML_PT07/blob/main/Funciones.ipynb

## ML model development in https://github.com/Abyzou1995/PI1_ML_PT07/blob/main/ML.ipynb


##API functions
@app.get("/")
def read_root():
    return {"Welcome to MLops project! By Angel Bello Merlo"}

@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero:str):## API function 1
    
    if (isinstance(genero,str)) :## parameter condition string
        genero=genero.lower()## parameter string to lower
        genero=unicodedata.normalize ('NFKD', genero).encode ('ascii', 'ignore').decode ('utf-8', 'ignore')## Filter accent mark
        
        jum1 = df_function_PlayTimeGenre[df_function_PlayTimeGenre["genres_lo"] == genero] ## filter dataset with the parameter

        if jum1.empty==True: ## condition if dataset is empty
            respuesta="No Data Avaliable"
            genero=genero.capitalize()
        else:
            respuesta = int(jum1.year.iloc[0])
            genero=jum1.genres.iloc[0]
        
    
        return {f'Año de lanzamiento con más horas jugadas para Género {genero}': respuesta}

"""TEST
PlayTimeGenre("ACTION")## Testing Genre Action
{'Año de lanzamiento con más horas jugadas para Género Action': 2012}
"""
@app.get('/UserForGenre/{genero}')
def UserForGenre(genero : str):## API function 2
    
    if (isinstance(genero,str)) :## parameter condition string
        genero=genero.lower()## parameter string to lower
        genero=unicodedata.normalize ('NFKD', genero).encode ('ascii', 'ignore').decode ('utf-8', 'ignore')## Filter accent mark
        
        jum = df_function_UserForGenre[df_function_UserForGenre["genres_lo"] == genero] ## filter dataset with the parameter
        
        if jum.empty==True: ## condition if dataset is empty
            usuario="No Data Available"
            respuesta="No Data Avaliable"
            genero1=genero.capitalize()
            
        else:
            genero1=jum.genres.iloc[0]
            dfu=jum.groupby(["user_id","genres","genres_lo"])["playtime_forever"].sum().reset_index().sort_values(by=["playtime_forever"],ascending=False).reset_index()
            usuario=dfu["user_id"].iloc[0]
            dfu2=jum.groupby(["user_id","genres","genres_lo","year"])["playtime_forever"].sum().reset_index().sort_values(by=["year"],ascending=False).reset_index()
            dfu3=dfu2[["year","playtime_forever"]][(dfu2["user_id"]==usuario) & (dfu2["genres_lo"]==genero)].reset_index()
            num=dfu3.shape[0]
            respuesta=[]
            for i in range(num-1):
                di={"Año": int(round(dfu3["year"].iloc[i])), "Horas": int(round(dfu3["playtime_forever"].iloc[i]))}
                respuesta.append(di)
                   
        return {f"Usuario con más horas jugadas para Género {genero1}" : usuario, "Horas jugadas":respuesta}

"""TEST
UserForGenre("Action")## Testing function2
{'Usuario con más horas jugadas para Género Action': 'Sp3ctre',
 'Horas jugadas': [{'Año': 2017, 'Horas': 722},
  {'Año': 2016, 'Horas': 493},
  {'Año': 2015, 'Horas': 5125},
  {'Año': 2014, 'Horas': 2178},
  {'Año': 2013, 'Horas': 2008},
  {'Año': 2012, 'Horas': 6305},
  {'Año': 2011, 'Horas': 2582},
  {'Año': 2010, 'Horas': 1301},
  {'Año': 2009, 'Horas': 1805},
  {'Año': 2008, 'Horas': 4},
  {'Año': 2007, 'Horas': 1880},
  {'Año': 2006, 'Horas': 15},
  {'Año': 2005, 'Horas': 356},
  {'Año': 2004, 'Horas': 2124},
  {'Año': 2003, 'Horas': 128},
  {'Año': 2002, 'Horas': 4},
  {'Año': 2001, 'Horas': 0},
  {'Año': 2000, 'Horas': 1177},
  {'Año': 1999, 'Horas': 1}]}
"""

@app.get('/UsersRecommend/{anio}')
def UsersRecommend( anio : int ):## API function 3
    
    if isinstance(anio,int):## parameter condition string
                
        jum = df_function_Recommends[(df_function_Recommends["year_posted"]==anio) & (df_function_Recommends["recommend"]==True) & (df_function_Recommends["sentiment_analysis"].isin([1,2]))] ## filter dataset with the parameter

        if jum.empty==True: ## condition if dataset is empty
            respuesta="No Data Avaliable"
        else:
            dfu4=jum.groupby(["app_name"])["id"].agg("count").reset_index()
            dfu4.sort_values(by="id",ascending=False,inplace=True)
            dfu4.reset_index(drop=True,inplace=True)
            respuesta=[]
            for i in range(3):
                di={f"Puesto {i+1}": dfu4["app_name"].iloc[i]}
                respuesta.append(di)
        
        return {f"Los juegos mas recomendados para el año {anio} son ": respuesta }

"""TEST
UsersRecommend( 2010 ) ## Testing function 3
{'Los juegos mas recomendados para el año 2010 son ': [{'Puesto 1': 'Team Fortress 2'},
  {'Puesto 2': 'Killing Floor'},
  {'Puesto 3': 'Alien Swarm'}]}
"""

@app.get('/UsersNotRecommend/{anio}')
def UsersNotRecommend( anio : int ):## API function 4
    
    if isinstance(anio,int):## parameter condition string
                
        jum = df_function_Recommends[(df_function_Recommends["year_posted"]==anio) & (df_function_Recommends["recommend"]==False) & (df_function_Recommends["sentiment_analysis"]==0)] ## filter dataset with the parameter

        if jum.empty==True: ## condition if dataset is empty
            respuesta="No Data Avaliable"
        else:
            dfu4=jum.groupby(["app_name"])["id"].agg("count").reset_index()
            dfu4.sort_values(by="id",ascending=False,inplace=True)
            dfu4.reset_index(drop=True,inplace=True)
            respuesta=[]
            for i in range(3):
                di={f"Puesto {i+1}": dfu4["app_name"].iloc[i]}
                respuesta.append(di)
        
        return {f"Los juegos menos recomendados para el año {anio} son ": respuesta }
    
"""TEST
UsersNotRecommend(2012)
{'Los juegos menos recomendados para el año 2012 son ': [{'Puesto 1': 'Red Faction®: Armageddon™'},
  {'Puesto 2': 'Team Fortress 2'},
  {'Puesto 3': "The Kings' Crusade"}]}
"""

@app.get('/Sentiment_Analysis/{anio}')
def Sentiment_Analysis( anio : int ):## API function 5
    
    if isinstance(anio,int):## parameter condition string
                
        jum = df_function_Sentiment[(df_function_Sentiment["year"]==anio)] ## filter dataset with the parameter

        if jum.empty==True: ## condition if dataset is empty
            respuesta="No Data Available"
        else:
            positivo=jum[jum["sentiment_analysis"]==2].shape[0]
            negativo=jum[jum["sentiment_analysis"]==0].shape[0]
            neutral=jum[jum["sentiment_analysis"]==1].shape[0]        
            respuesta={"Positivo":positivo,"Neutral":neutral,"Negativo":negativo}
        return {f"El analisis de sentimiento para el año de lanzamiento {anio} es ": respuesta }

"""TEST

Sentiment_Analysis( 2010 )## Testing function N° 5
{'El analisis de sentimiento para el año de lanzamiento 2010 es ': {'Positivo': 947,
  'Neutral': 1854,
  'Negativo': 255}}
"""

@app.get('/Game_Recommendation/{id}')
def Game_Recommendation( id : int ):## API function ML
    
    if isinstance(id,int):## parameter condition string
        indices = df_function_Item_Item[["item_id", "index"]]    
        tfidf=TfidfVectorizer(stop_words="english",max_features=10000)## Setting tf-idf vector
        tfidf_matrix=tfidf.fit_transform(df_function_Item_Item["features"])## Setting tf-idf vectorizer with data
        cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)## Training model with the given data   
        idx = indices[indices["item_id"]==id]## filter dataset with the parameter
        if idx.empty== True:## condition if dataset is empty
            recommendations=["No data available"]
        else:
            idy = idx["index"].iloc[0]## Search index
            sim_score = list(enumerate(cosine_sim[idy]))## Setting similarity
            sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)## Sorting results by score
            sim_score = sim_score[1:6]## Getting top score 5 movies
            movies_index = [i[0] for i in sim_score] ## Finding names
            recommendations = list(df_function_Item_Item['app_name'].iloc[movies_index])## Making the list
    return {'ID de juego':id,'lista recomendada': recommendations}
        
"""TEST
Game_Recommendation(10)## Testing ML function
{'Id de juego': 10,
 'lista recomendada': ['Counter-Strike: Condition Zero',
  'Counter-Strike: Source',
  'Counter-Strike: Global Offensive',
  'Insurgency',
  'Team Fortress Classic']}
"""

       
