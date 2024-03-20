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

df_function_PlayTimeGenre=pd.read_parquet("Dataset/Function_Dataset/F1.parquet")
df_function_UserForGenre=pd.read_parquet("Dataset/Function_Dataset/F2.parquet")
df_function_Recommends=pd.read_parquet("Dataset/Function_Dataset/F3-4-5.parquet")
df_function_Sentiment=pd.read_parquet("Dataset/Function_Dataset/F5.parquet")
df_function_Item_Item=pd.read_parquet("Dataset/Function_Dataset/MLF1.parquet")
##Functions development in https://github.com/Abyzou1995/PI01_DATA10_MLops_HENRY/blob/main/FunctionAPI_MLops.ipynb

## ML model development in https://github.com/Abyzou1995/PI01_DATA10_MLops_HENRY/blob/main/ModelML_MLops.ipynb


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
    return {'titulo':id,'lista recomendada': recommendations}
        


       