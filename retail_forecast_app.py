#------------------------------
#Librerias
#------------------------------
import pandas as pd
import numpy as np
import plotly.express as px                                                                                                                                                                                    
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pyodbc
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

#---------------------------
#Cargar logo de la empresa
#---------------------------
UBICACION_IMAGEN = r"ruta/a/tu/imagen.png" #Cambiar ubicación de archivo
try:
    logo=Image.open(UBICACION_IMAGEN)
    st.image(logo,width=150)
except FileNotFoundError:
    pass


#------------------------------------------------------------------------
#Titulo del proyecto
#------------------------------------------------------------------------
st.title("**Estimación de ventas y tendencias de mercado de GM**")


#-------------------------------------------------------------
#Cargar datos de venta
#-------------------------------------------------------------
def cargar_datos_venta():
    """
    En el proyecto original los datos fueron obtenidos desde una fuente corporativa mediante SQL.
    Por razones de confidencialidad, los datos originales y la conexión corporativa no forman parte de este repositorio.
    """
    return pd.read_csv("data/datos_venta.csv")

#Ejecutar consulta
df_operacion=cargar_datos_venta() 

#Reemplazar nulos por cero
df_operacion=df_operacion.fillna(0) 

#Desactivar notacion cientifica
pd.set_option('display.float_format', '{:,.2f}'.format) 

#Transformar fecha a formato datetime
df_operacion['Fecha']=pd.to_datetime(df_operacion['Fecha']) 

#Agrupar ventas diarias de forma agregada
df_diario=df_operacion.groupby('Fecha',as_index=False)['Venta_Retail'].sum()


#-----------------------------------------------
# Cargar Datos del Banco Central de Chile
#-----------------------------------------------
#Leer archivo de datos
UBICACION_DATOS_BCCH = r"ruta/a/tu/archivo_banco_central.csv" #Cambiar ubicación de archivo
df_bcch=pd.read_csv(UBICACION_DATOS_BCCH) 

#Transformar fecha a formato datetime
df_bcch["Fecha"]=pd.to_datetime(df_bcch["Fecha"]) 

#Crear una copia del archivo de BCCH
df_bc_crecimiento=df_bcch.copy() 

#Seleccionar columnas relevantes del archivo de BCCH para la prediccion
df_bcch2=df_bcch[["Fecha","Indice de Transacciones de Comercio Minorista"]] 

#Unir ventas e indice economico
df_venta_ivcm=pd.merge(df_diario,df_bcch2,on="Fecha",how="left")

#Ordenar por fecha y reordenar indice
df_venta_ivcm=df_venta_ivcm.sort_values("Fecha").reset_index(drop=True) 

#Asignar cero a valores nulos
df_venta_ivcm=df_venta_ivcm.fillna(0) 

#Crear copia del archivo a pronosticar
df_comportamiento=df_venta_ivcm.copy() 

#Renombrar columnas
df_venta_ivcm.rename(columns={'Fecha': 'ds'}, inplace=True) #Fecha por ds
df_venta_ivcm.rename(columns={'Venta_Retail': 'y'}, inplace=True) #Venta por y
df_venta_ivcm.rename(columns={'Indice de Transacciones de Comercio Minorista': 'indice'}, inplace=True) #IVCM por indice


#-----------------------------------------
#Eventos 
#-----------------------------------------
#Feriados irrenunciables
feriados=[
    (1, 1), #1 de enero   
    (5, 1), #1 de mayo    
    (9, 18), #18 de septiembre   
    (9, 19), #19 de septiembre   
    (12, 25) #25 de diciembre  
]

año_minimo=df_venta_ivcm['ds'].dt.year.min() #Extraer primer año 
año_maximo = df_venta_ivcm['ds'].dt.year.max() #Extraer ultimo año

fechas_siempre_cero=[] #Lista para guardar las fecha

#El bucle recorre desde el primer al ultimo año
for año in range(año_minimo,año_maximo+2):  
    for mes, dia in feriados:
        fechas_siempre_cero.append(pd.Timestamp(year=año,month=mes,day=dia)) #Crear fecha 

fechas_siempre_cero=pd.to_datetime(fechas_siempre_cero) #Transformar a formate datetime las fechas creadas

#DataFrame de los dias feriados  
eventos=pd.DataFrame({
    'holiday':'feriado', #Nombre
    'ds':fechas_siempre_cero, #Fechas
    'lower_window':0, #Efecto dias antes del feriado
    'upper_window':1 #Efectos dias despues del feriado 
})

#Colocar en cero esas fechas
df_venta_ivcm.loc[df_venta_ivcm['ds'].isin(fechas_siempre_cero), 'y']=0

#Venta alta en diciembre
df_venta_ivcm['mes'] = df_venta_ivcm['ds'].dt.month #Crear una columna para el mes
df_venta_ivcm['venta_alta'] = (df_venta_ivcm['mes'] == 12).astype(int) #Crear columna binaria (se asigna 1 cuando la columnas es igual a 12)
df_venta_ivcm.drop(columns=['mes'], inplace=True) #Eliminar la columna mes


#---------------------------------------------------
#Secciones y filtro
#---------------------------------------------------
horizonte=st.slider("Selecciona los días a predecir:",min_value=1,max_value=365,value=30,step=1,key="slider_horizonte") #Filtro de horizonte de prediccion

seccion1,seccion2=st.tabs(["**Comportamiento del Consumidor**","**Oportunidades de Mercado**"]) #Numero de pestañas


#---------------------------------------------------
#Pestaña 1: comportamiento del consumidor
#---------------------------------------------------
with seccion1:
    #-------------------------------
    #Preparar DataFrame
    #-------------------------------
    df_indice=df_venta_ivcm[['ds','indice']].copy() #Crear copia de archivo a pronosticar para predecir el IVCM
    df_indice.rename(columns={'indice':'y'},inplace=True) #Renombre IVCM por y

    #----------------------------
    #Modelo predictivo
    #----------------------------
    #Seleccionar parametros
    parametros_indice={ 
        "weekly_seasonality":True, #Activar estacionalidad semanal 
        "yearly_seasonality":True, #Activar estacionalidad anual
        "daily_seasonality":False, #Desactivar estacionalidad diaria
        "holidays_prior_scale":5, #Influencia de los feriados en el modelo
        "seasonality_mode":"multiplicative", #Tipo de estacionalidad
        "seasonality_prior_scale":5, #Flexibilidad de la estacionalidad
        "changepoint_prior_scale":0.01, #Sensibilidad a cambio de tendencia 
        "interval_width":0.95, #Intervalo de confianza
        "holidays":eventos #DataFrame con los feriados
    }

    modelo_indice = Prophet(**parametros_indice) #Crear modelo
    modelo_indice.add_seasonality(name='monthly',period=30.5,fourier_order=5) #Activar estacionalidad mensual
    modelo_indice.fit(df_indice) #Entrenar modelo


    #-------------------------------------------------------
    #Numero de dias a predecir incluyendo ceros historicos
    #-------------------------------------------------------
    #El IVCM no contiene la misma cantidad de filas que el DataFrame de ventas, por ende se deben predecir las observaciones necesarias para que ambas columnas esten a la par y ademas se debe pronosticar el horizonte de 365 dias
    total_cero_indice=df_indice.loc[df_indice['y']==0,'ds'] #Filtrar las fechas en las cuales y es igual a cero

    ultima_fecha=df_indice['ds'].max() #Obtener la ultima fecha del DataFrame a pronosticar

    fechas_predecir_indice=pd.date_range(start=ultima_fecha+pd.Timedelta(days=1),periods=horizonte,freq='D') #Crear DataFrame de fechas a pronosticar en base al horizonte de 365 dias

    fechas_predecir_indice=pd.Series(fechas_predecir_indice) #Transformar a series las fechas a predecir

    total_fechas_predecir_indice=pd.concat([total_cero_indice,fechas_predecir_indice]).sort_values().unique() #Se juntan las fechas con valor y igual a cero, pero tambien las fechas del horizonte de 365 dias

    df_predecir_indice = pd.DataFrame({'ds': total_fechas_predecir_indice}) #Crear DataFrame del total de fechas a predecir del IVCM 

    #-------------------------------
    #Predecir IVCM
    #-------------------------------
    pronostico_indice=modelo_indice.predict(df_predecir_indice)

    #------------------------------------------------------
    #Mapear predicciones para reemplazar ceros historicos
    #------------------------------------------------------
    #Reemplazar ceros históricos para nivelar la cantidad de observaciones de venta con el IVCM
    mapa_pronostico_indice = pronostico_indice.set_index('ds')['yhat'] #Mapa de fecha y valor predicho
    df_venta_ivcm['indice']=df_venta_ivcm['indice'].mask(df_venta_ivcm['indice']==0,df_venta_ivcm['ds'].map(mapa_pronostico_indice)) #Reemplazar los ceros por los valores predichos 

    #---------------------------------------------------------
    #Combinar DataFrame historico con futuro para el gráfico
    #---------------------------------------------------------
    no_esta_historico=pronostico_indice[pronostico_indice['ds']>df_venta_ivcm['ds'].max()][['ds','yhat','yhat_lower','yhat_upper']] #Considerar solo las fechas futuras que no estan en el historico
    no_esta_historico.rename(columns={'yhat':'indice'},inplace=True) #Renombrar prediccion
    df_grafico_hp_indice=pd.concat([df_venta_ivcm[['ds','indice']],no_esta_historico]).reset_index(drop=True) #Crear DataFrame para grafico historico con prediccion

    #--------------------------------------
    #Correlacion de venta con IPC e IMACEC
    #--------------------------------------
    df_bc_ipc_imacec=df_bcch[["Fecha","ipc","imacec"]] #Crear DataFrame para IPC e IMACEC 
    df_mensual=df_diario[["Fecha","Venta_Retail"]] #Crear DataFrame para la venta mensual 

    #Agrupar por mes la venta (los datos se agrupan el primero de cada mes)
    df_mensual["Añomes"]=df_mensual["Fecha"].dt.to_period("M").dt.to_timestamp()
    df_mensual=df_mensual.groupby("Añomes",as_index=False)["Venta_Retail"].sum()
    df_mensual=df_mensual.rename(columns={"Añomes":"Fecha"})
    
    #DataFrame final para la correlacion de la venta mensual con IPC e IMACEC
    df_mensual_ipc_imacec = pd.merge(df_bc_ipc_imacec,df_mensual[["Fecha","Venta_Retail"]],on="Fecha",how="left")

    #Titulo de la seccion en la aplicacion
    st.subheader("Relación entre la Venta e Indicadores Económicos") 

    #Seleccionar rango de fechas para el filtro
    fecha_inicio_correlacion_ipc_imacec,fecha_fin_correlacion_ipc_imacec=st.date_input(
        'Seleccionar rango de fechas:',
        value=(df_mensual_ipc_imacec['Fecha'].min(), df_mensual_ipc_imacec['Fecha'].max()))

    #Filtro de fechas para el DataFrame
    df_filtrado_corr=df_mensual_ipc_imacec[(df_mensual_ipc_imacec['Fecha']>=pd.to_datetime(fecha_inicio_correlacion_ipc_imacec))&(df_mensual_ipc_imacec['Fecha']<=pd.to_datetime(fecha_fin_correlacion_ipc_imacec))]

    #Mantener las observaciones necesarias
    df_correlacion_ipc_imacec=df_filtrado_corr[["Venta_Retail","ipc","imacec"]].dropna()

    #Correlacion de IPC e IMACEC
    correlacion_ipc=df_correlacion_ipc_imacec["Venta_Retail"].corr(df_correlacion_ipc_imacec["ipc"],method="spearman")
    correlacion_imacec=df_correlacion_ipc_imacec["Venta_Retail"].corr(df_correlacion_ipc_imacec["imacec"],method="spearman")

    #Clasificar metricas
    def clasificar_correlacion(x):
        '''Clasificar la correlacion para determinar el grado de influencia de la economia en la venta'''
        correlacion=abs(x) #Valor absoluto  
        if correlacion<0.2:
            return "La Venta Depende de las Acciones Comerciales"
        elif 0.2 <=correlacion<0.4:
            return "La Venta Depende de las Acciones Comerciales"
        elif 0.4 <=correlacion<0.6:
            return "La Venta tiene Moderada Dependencia del Contexto Macroeconómico"
        elif 0.6 <=correlacion<0.8:
            return "La Venta tiene Alta Dependencia del Contexto Macroeconómico"
        else:
            return "La Venta tiene Fuerte Dependencia del Contexto Macroeconómico"

    categoria_ipc=clasificar_correlacion(correlacion_ipc)
    categoria_imacec=clasificar_correlacion(correlacion_imacec)

    #Visualizar la correlacin en la aplicacion
    columna_ipc, columna_imacec=st.columns(2) #Crear columnas
    #Mostrar metricas 
    columna_ipc.metric(
        "**Correlación Venta vs IPC**",
        value=round(correlacion_ipc, 4), #Numero
        delta=categoria_ipc #Clasificacion
    )

    columna_imacec.metric(
        "**Correlación Venta vs IMACEC**",
        value=round(correlacion_imacec, 4), #Numero
        delta=categoria_imacec #Clasificacion
    )

    #---------------------------------------
    #Grafico de doble eje
    #--------------------------------------
    #Preparar DataFrame
    df_comportamiento.rename(columns={'Indice de Transacciones de Comercio Minorista':'indice'}, inplace=True) #Renombrar IVCM
    df_comportamiento = df_comportamiento[df_comportamiento['indice'] != 0].reset_index(drop=True) #Filtrar datos historicos del IVCM

    #Titulo del grafico
    st.subheader("**Venta vs Índice de Transacciones de Comercio Minorista (IVCM)**")

    #Crear figura con doble eje 
    figura_doble_eje=make_subplots(specs=[[{"secondary_y": True}]])

    #Crear linea de la venta (eje izquierdo) 
    figura_doble_eje.add_trace(
        go.Scatter(
            x=df_comportamiento['Fecha'], #Eje x
            y=df_comportamiento['Venta_Retail'], #Eje izquierdo (y)
            mode='lines', #Tipo de grafico
            name='Venta (CLP)', #Nombre de la serie 
            line=dict(color="#1b88e7",width=2), #Color y dimension de la linea 
            hovertemplate="Venta: %{y}<extra></extra>" #Configuracion para visualizar de forma estrategica las observaciones
        ),
        secondary_y=False #Se aplica al eje izquierdo
    )

    #Crear linea del IVCM (eje derecho) 
    figura_doble_eje.add_trace(
        go.Scatter(
            x=df_comportamiento['Fecha'], #Eje X
            y=df_comportamiento['indice'], #Eje derecho (y)
            mode='lines', #Tipo de grafico
            name='IVCM', #Nombre de la serie 
            line=dict(color="#eba800",width=2) #Color y dimension de la linea 
        ),
        secondary_y=True #Se aplica al eje derecho
    )

    #Configuracion general
    figura_doble_eje.update_layout(
        xaxis_title='Fecha', #Titulo de eje x
        template='plotly_white', #Diseño del grafico
        hovermode='x unified' #Mostrar valores de la serie de datos en una fecha especifica
    )

    #Configurar el eje de venta
    figura_doble_eje.update_yaxes(
        title_text="Venta (CLP)", #Titulo
        tickformat="", #No abreviar numeros              
        separatethousands=False, #No usar la escala numerica automatica de la libreria  
        exponentformat="none", #No usar notacion cientifica     
        showexponent="none", #No mostrar exponente
        secondary_y=False #Aplicar configuracion
    )

    #Configurar eje de IVCM
    figura_doble_eje.update_yaxes(
        title_text="IVCM", #Titulo del eje
        secondary_y=True #Aplicar configuracion
    )

    #Mostrar grafico de doble eje en la aplicacion
    st.plotly_chart(figura_doble_eje,use_container_width=True)


    #----------------------------------
    #Grafico de prediccion para IVCM
    #----------------------------------
    st.subheader("**Indice de Transacciones de Comercio Minorista (IVCM) - Datos Históricos y Predicción**") #Titulo de la seccion

    figura_prediccion_indice = go.Figure() #Crear figura para el grafico de prediccion del IVCM

    #Configurar linea histórica
    figura_prediccion_indice.add_trace(go.Scatter(
        x=df_grafico_hp_indice['ds'], #Fechas historicas
        y=df_grafico_hp_indice['indice'], #Valores historicos
        mode='lines', #TIpo de grafico
        name='Histórico', #Nombre de la serie 
        line=dict(width=2,color="#1b88e7") #Color y dimension de la linea 
    ))

    #Configurar prediccion 
    registros_pronostico=df_grafico_hp_indice['yhat_upper'].notna() #Obtener registros del pronostico

    figura_prediccion_indice.add_trace(go.Scatter(
        x=df_grafico_hp_indice.loc[registros_pronostico,'ds'], #Fechas de la prediccion
        y=df_grafico_hp_indice.loc[registros_pronostico,'indice'], #Registros de la prediccion
        mode='lines', #Tipo de grafico
        name='Predicción', #Nombre 
        line=dict(dash='dot',width=3,color="#eba800") #Configurar linea y color
    ))

    #Limite superior del intervalo de confianza
    figura_prediccion_indice.add_trace(go.Scatter(
        x=df_grafico_hp_indice.loc[registros_pronostico,'ds'], #Fecha de prediccion
        y=df_grafico_hp_indice.loc[registros_pronostico,'yhat_upper'], #Limite superior de intervalo de confianza
        mode='lines', #Tipo de grafico 
        line=dict(width=0, color='rgba(0,0,0,0)'), #Configuracion de linea y color
        showlegend=False #No esta en la leyenda
    ))

    #Limite inferior del intervalo de confianza
    figura_prediccion_indice.add_trace(go.Scatter(
        x=df_grafico_hp_indice.loc[registros_pronostico,'ds'], #Fecha de prediccion
        y=df_grafico_hp_indice.loc[registros_pronostico,'yhat_lower'], #Limite inferior de intervalo de confianza
        mode='lines', #Tipo de grafico 
        fill='tonexty', #Completar color
        name='Intervalo de confianza', #Nombre
        line=dict(width=0), ##Configuracion de linea y color
        fillcolor='rgba(200,200,200,0.3)'  #Color
    ))

    figura_prediccion_indice.update_layout(
        xaxis_title='Fecha', #Titulo de eje x
        yaxis_title='Valor', #Titulo de eje y
        template='plotly_white', #Diseño del grafico
        hovermode='x unified' #Mostrar valores de la serie de datos en una fecha especifica
    )

    figura_prediccion_indice.update_yaxes(tickformat=",") #Mostrar numeros en su escala original

    st.plotly_chart(figura_prediccion_indice, use_container_width=True) #Visualizacion en la aplicacion


    #----------------------------------------
    #Entrenar modelo de validacion
    #----------------------------------------
    #Procesar DataFrame
    df_bcch.rename(columns={'Indice de Transacciones de Comercio Minorista':'indice'}, inplace=True) #Renombrar IVCM
    df_bcch.rename(columns={'Fecha':'ds'},inplace=True) #Renombrar fecha
    df_modelo_indice_validacion=df_bcch[['ds','indice']].copy() #Copiar DataFrame de BCCH
    df_modelo_indice_validacion.rename(columns={'indice':'y'}, inplace=True) #Renombrar IVCM

    #Modelo
    modelo_indice_validacion=Prophet(**parametros_indice) #Crear modelo
    modelo_indice_validacion.add_seasonality(name='monthly',period=30.5,fourier_order=5) #Estacionalidad mensual

    #Conjuntos de datos
    ventana_indice_validacion=60 #Ventana rodante de 60 dias
    entrenamiento_indice_validacion=df_modelo_indice_validacion.iloc[:-ventana_indice_validacion].copy() #Entrenamiento
    prueba_indice_validacion=df_modelo_indice_validacion.iloc[-ventana_indice_validacion:].copy() #Prueba

    #Entrenar modelo
    modelo_indice_validacion.add_seasonality(name='monthly',period=30.5,fourier_order=5) #Estacionalidad mensual
    modelo_indice_validacion.fit(entrenamiento_indice_validacion) #Entrenar modelo con datos historicos

    #-----------------------------------
    #Validacion de 60 dias
    #-----------------------------------
    fechas_futuras_validacion_indice=prueba_indice_validacion[['ds']] #Fechas a predecir
    prediccion_validacion_indice=modelo_indice_validacion.predict(fechas_futuras_validacion_indice) #Pronostico de las fechas a predecir

    #-------------------------------
    # Grafico de validacion
    #-------------------------------
    #Titulo se la seccion
    st.subheader("**Validación de Predicción de los últimos 60 días del Indice de Transacciones de Comercio Minorista (IVCM)**")

    #Preparar DataFrame para el grafico de validacion
    df_grafico_validacion_indice=prueba_indice_validacion[['ds','y']].merge(prediccion_validacion_indice[['ds','yhat','yhat_lower','yhat_upper']],on='ds',how='left')

    #Crear figura para el grafico de validacion
    grafico_validacion_indice=go.Figure()

    #Linea historica
    grafico_validacion_indice.add_trace(go.Scatter(
        x=df_grafico_validacion_indice['ds'], #Fecha historica
        y=df_grafico_validacion_indice['y'], #Serie historica
        mode='lines', #Tipo de grafico
        name='Histórico', #Nombre
        line=dict(width=2,color="#1b88e7") #Configuracion de la linea
    ))

    #Linea de prediccion
    seleccionar_prediccion_indice_validacion=df_grafico_validacion_indice['yhat'].notna() #Seleccion de observaciones predichas

    grafico_validacion_indice.add_trace(go.Scatter(
        x=df_grafico_validacion_indice.loc[seleccionar_prediccion_indice_validacion,'ds'], #Fecha
        y=df_grafico_validacion_indice.loc[seleccionar_prediccion_indice_validacion,'yhat'], #Prediccion
        mode='lines', #Tipo de grafico
        name='Predicción', #Nombre
        line=dict(dash='dot', width=3, color="#eba800") #Configuracion de la linea
    ))

    #Intervalo de confianza superior
    grafico_validacion_indice.add_trace(go.Scatter(
        x=df_grafico_validacion_indice.loc[seleccionar_prediccion_indice_validacion, 'ds'], #Fecha
        y=df_grafico_validacion_indice.loc[seleccionar_prediccion_indice_validacion, 'yhat_upper'], #Registro
        mode='lines', #Tipo de grafico
        line=dict(width=0, color='rgba(0,0,0,0)'), #Configuracion de la linea
        showlegend=False #No esta en la leyenda
    ))

    #Intervalo de confianza inferior
    grafico_validacion_indice.add_trace(go.Scatter(
        x=df_grafico_validacion_indice.loc[seleccionar_prediccion_indice_validacion, 'ds'],
        y=df_grafico_validacion_indice.loc[seleccionar_prediccion_indice_validacion, 'yhat_lower'],
        mode='lines', #Tipo de grafico
        fill='tonexty', #Completar color
        name='Intervalo de confianza', #Nombre
        line=dict(width=0), #Configuracion de la linea
        fillcolor='rgba(200,200,200,0.3)'  ##Configuracion de la linea
    ))

    #Configuracion del grafico
    grafico_validacion_indice.update_layout(
        xaxis_title='Fecha', #Fecha
        yaxis_title='Valor', #Registro
        template='plotly_white', #Configuracion de fondo del grafico
        hovermode='x unified' #Visualizacion de los registros
    )

    grafico_validacion_indice.update_yaxes(tickformat=",") #Formato de numeros

    st.plotly_chart(grafico_validacion_indice, use_container_width=True) #Visualizar en la aplicacion

    #-------------------------------
    #Metricas de validacion
    #-------------------------------
    mape_indice_validacion=mean_absolute_percentage_error(prueba_indice_validacion['y'],prediccion_validacion_indice['yhat']) #MAPE
    rmse_indice_validacion=np.sqrt(mean_squared_error(prueba_indice_validacion['y'],prediccion_validacion_indice['yhat'])) #RMSE

    columnamape_indice_validacion, columnarmse_indice_validacion = st.columns(2)

    columnamape_indice_validacion.metric(
        "**Error Porcentual (MAPE):**",
        value=f"{mape_indice_validacion*100:.4f}%" #Valor
    )

    columnarmse_indice_validacion.metric(
        "**Desviación en Unidades (RMSE):**",
        value=round(rmse_indice_validacion, 4) #Valor
    )





with seccion2:
    #------------------------
    #Modelo de venta
    #------------------------
    parametros_venta = {
        "weekly_seasonality": True, #Activar estacionalidad semanal 
        "yearly_seasonality": True, #Activar estacionalidad anual
        "daily_seasonality": False, #Desactivar estacionalidad diaria
        "holidays_prior_scale": 10, #Influencia de los feriados en el modelo
        "seasonality_mode": "multiplicative", #Tipo de estacionalidad
        "seasonality_prior_scale": 15.0, #Flexibilidad de la estacionalidad
        "changepoint_prior_scale": 0.049, #Sensibilidad a cambio de tendencia 
        "interval_width": 0.95, #Intervalo de confianza
        "holidays": eventos #DataFrame con los feriados
    }

    modelo_venta = Prophet(**parametros_venta) #Crear modelo
    modelo_venta.add_regressor('indice') #Regresor ICVM
    modelo_venta.add_regressor('venta_alta') #Regresor binario que indica cuando la venta es alta
    modelo_venta.add_seasonality(name='monthly',period=30.5,fourier_order=5) #Añadir estacionalidad mensual
    modelo_venta.fit(df_venta_ivcm[['ds','y','indice','venta_alta']]) #Entrenar modelo

    #------------------------
    #Fechas a predecir
    #------------------------
    ultima_fecha_venta=df_venta_ivcm['ds'].max() #Ultima fecha del DataFrame

    fechas_predecir_ventas=pd.date_range(start=ultima_fecha_venta+pd.Timedelta(days=1),periods=horizonte,freq='D') #Rango de fechas a predecir

    df_fechas_predecir_ventas=pd.DataFrame({'ds':fechas_predecir_ventas}) #Crear DataFrame con las fechas que se deben predecir

    df_fechas_predecir_ventas=df_fechas_predecir_ventas.merge(df_grafico_hp_indice[['ds','indice']],on='ds',how='left') #Unir la fecha con el indice para crear el DataFrame que se debe predecir

    df_fechas_predecir_ventas['venta_alta']=(df_fechas_predecir_ventas['ds'].dt.month==12).astype(int) #Crear columna binaria de venta alta al DataFrame que se debe predecir

    #---------------------------
    #Prediccion de la venta
    #---------------------------
    predecir_ventas=modelo_venta.predict(df_fechas_predecir_ventas)

    #Asegurar formato datetime
    predecir_ventas['ds']=pd.to_datetime(predecir_ventas['ds'])
    fechas_cero=pd.to_datetime(fechas_siempre_cero) 

    #Identificar feriados irrenunciables
    feriados_venta_cero=predecir_ventas['ds'].isin(fechas_cero)

    #Aplicar venta cero en todos los feriados irrenunciables
    predecir_ventas.loc[feriados_venta_cero,['yhat','yhat_lower','yhat_upper']]=0

    columnas_venta_cero=['yhat','yhat_lower','yhat_upper'] #Columnas con venta nula 

    predecir_ventas[columnas_venta_cero]=predecir_ventas[columnas_venta_cero].round(0).astype(int) #Asegurar tipo de datos a entero (sin decimales)

    #------------------------------------------
    #Grafico de venta historica y prediccion
    #------------------------------------------
    figura_hp_venta = go.Figure()

    #Linea historica
    figura_hp_venta.add_trace(go.Scatter(
        x=df_venta_ivcm['ds'], #Fecha
        y=df_venta_ivcm['y'], #Registros
        mode='lines', #Tipo de grafico
        name='Histórico', #Nombre
        line=dict(width=2,color="#1b88e7"), #Configurar linea
        hovertemplate="Histórico: %{y}<extra></extra>" #Ver registros
    ))

    #Lines de prediccion
    figura_hp_venta.add_trace(go.Scatter(
        x=predecir_ventas['ds'], #Fecha
        y=predecir_ventas['yhat'], #Registros
        mode='lines', #Tipo de grafico
        name='Predicción', #Nombre
        line=dict(dash='dot', width=3, color="#eba800"), #Configurar linea
        hovertemplate="Predicción: %{y}<extra></extra>" #Ver registros
    ))

    #Intervalo de confianza superior
    figura_hp_venta.add_trace(go.Scatter(
        x=predecir_ventas['ds'], #Fecha
        y=predecir_ventas['yhat_upper'], #Registros
        mode='lines', #Tipo de grafico
        line=dict(width=0), #Configurar linea
        showlegend=False #No se ve en la leyenda
    ))

    #Configuracion general del grafico
    figura_hp_venta.update_layout(
        hovermode='x unified', #Ver registros
        xaxis_title='Fecha', #Nombre eje y
        yaxis_title='Venta (CLP)', #Nombre eje y
        template='plotly_white' #Estilo del grafico 
    )

    #Intervalo de confianza inferior
    figura_hp_venta.add_trace(go.Scatter(
        x=predecir_ventas['ds'], #Fecha
        y=predecir_ventas['yhat_lower'], #Registros
        mode='lines', #Tipo de grafico
        fill='tonexty', #Completar linea
        name='Intervalo de confianza', #Nombre
        line=dict(width=0), #Configuracion de la linea
        fillcolor='rgba(200,200,200,0.3)', #Color
        hovertemplate="Inferior: %{y}<extra></extra>" #Ver registros
    ))

    #Configurar eje y
    figura_hp_venta.update_yaxes(
        tickformat="", #Mostrar numero real
        separatethousands=False, #Desactivar formato automatico
        exponentformat="none", #Desactivar notacion cientifica
        showexponent="none" #Desactivar exponente
    )

    st.subheader("Ventas de GM - Datos Históricos y Predicción") #Titulo de la seccion
    st.plotly_chart(figura_hp_venta, use_container_width=True) #Visualizar en la aplicacion

    #--------------------------------------------
    #Mostrar tabla para descargar prediccion
    #--------------------------------------------
    st.subheader("Tabla de Predicciones de Ventas") #Titulo de la seccion

    #Columnas a incluir en la tabla para descargar
    columnas_descargar=['ds','yhat','yhat_lower','yhat_upper'] 
    df_prediccion_descargar=predecir_ventas[columnas_descargar]

    #Renombrar columnas
    df_prediccion_descargar.rename(columns={'ds': 'Fecha',
                                   'yhat':'Prediccion',
                                   'yhat_upper':'Limite superior',
                                   'yhat_lower':'Limite inferior'
                                   }, inplace=True)
    
    st.dataframe(df_prediccion_descargar) #Visaulizar en la aplicacion


    #--------------------------------------
    #Validar prediccion
    #--------------------------------------
    #Conjuntos de datos
    ventana_validacion_venta=60 

    df_entrenamiento_ventas=df_venta_ivcm.iloc[:-ventana_validacion_venta].copy() #Datos de entrenamiento 
    df_prueba_ventas=df_venta_ivcm.iloc[-ventana_validacion_venta:].copy() #Datos de prueba

    #Modelo
    modelo_validacion_ventas=Prophet(**parametros_venta) #Crear modelo
    modelo_validacion_ventas.add_regressor('indice') #Añadir regresor indice
    modelo_validacion_ventas.add_regressor('venta_alta') #Añadir regresor binario de venta alta
    modelo_validacion_ventas.add_seasonality(name='monthly',period=30.5,fourier_order=5) #Añadir estacionalidad mensual
    modelo_validacion_ventas.fit(df_entrenamiento_ventas[['ds','y','indice','venta_alta']]) #Entrenar modelo

    #Construir DataFrame a predecir
    fechas_prueba_modelovalidacion = pd.DataFrame({'ds': df_prueba_ventas['ds']}) #DataFrame para predecir las fechas del conjunto de prueba
    fechas_prueba_modelovalidacion=fechas_prueba_modelovalidacion.merge(df_grafico_hp_indice[['ds','indice']],on='ds',how='left') #Añdir IVCM al DataFrame de pronostico
    fechas_prueba_modelovalidacion['venta_alta']=(fechas_prueba_modelovalidacion['ds'].dt.month==12).astype(int) #Añdir regresor de venta alta al DataFrame

    #Prediccion
    prediccion_ventas_validacion=modelo_validacion_ventas.predict(fechas_prueba_modelovalidacion) #Predecir venta

    #Colocar en cero todos los feriados irrenunciables
    prediccion_ventas_validacion['ds']=pd.to_datetime(prediccion_ventas_validacion['ds']) #Aplicar formato datetime
    feriados_ventacero_validacion = prediccion_ventas_validacion['ds'].isin(fechas_cero) #Seleccionar feriados irrenunciables 
    prediccion_ventas_validacion.loc[feriados_ventacero_validacion,['yhat','yhat_lower','yhat_upper']]=0 #Colocar cero de venta a los feriados irrenunciables

    #Obtener metricas de error
    y_real_prueba=df_prueba_ventas['y'].values #Valor real
    y_predicho_prueba=prediccion_ventas_validacion['yhat'].values #Valor predicho

    mape_validacion_venta=mean_absolute_percentage_error(y_real_prueba,y_predicho_prueba) #MAPE
    rmse_validacion_venta=np.sqrt(mean_squared_error(y_real_prueba,y_predicho_prueba)) #RMSE


    #Graficar validacion
    st.subheader("Validación de Predicción de Ventas de los últimos 60 días de GM") #Titulo de la seccion

    figura_prediccion_validacion_venta=go.Figure() #Crear grafico

    #Linea historica
    figura_prediccion_validacion_venta.add_trace(go.Scatter(
        x=df_prueba_ventas['ds'], #Fecha
        y=df_prueba_ventas['y'], #Registros
        mode='lines', #Tipo de grafico
        name='Real', #Nombre
        line=dict(width=2,color="#1b88e7"), #Configuracion de linea
        hovertemplate="Real: %{y}<extra></extra>" #Ver registros
    ))

    #Linea de prediccion
    figura_prediccion_validacion_venta.add_trace(go.Scatter(
        x=prediccion_ventas_validacion['ds'], #Fecha
        y=prediccion_ventas_validacion['yhat'], #Registros
        mode='lines', #Tipo de grafico
        name='Predicción', #Nombre
        line=dict(dash='dot', width=3,color="#eba800"), #Configuracion de linea
        hovertemplate="Predicción: %{y}<extra></extra>" #Ver registros
    ))

    #Intervalo de confianza superior
    figura_prediccion_validacion_venta.add_trace(go.Scatter(
        x=prediccion_ventas_validacion['ds'], #Fecha
        y=prediccion_ventas_validacion['yhat_upper'], #Registros
        mode='lines', #Tipo de grafico
        line=dict(width=0), #Configuracion de linea
        showlegend=False #No esta en la leyenda
    ))

    #Intervalo de confianza inferior
    figura_prediccion_validacion_venta.add_trace(go.Scatter(
        x=prediccion_ventas_validacion['ds'], #Fecha
        y=prediccion_ventas_validacion['yhat_lower'], #Registros
        mode='lines', #Tipo de grafico
        fill='tonexty', #Completar grafico
        name='Intervalo de confianza', #Nombre
        line=dict(width=0), #Configuracion de linea
        fillcolor='rgba(200,200,200,0.3)', #Color
        hovertemplate="Inferior: %{y}<extra></extra>" #Ver registros
    ))

    #Configuracion general
    figura_prediccion_validacion_venta.update_layout(
        hovermode='x unified', #Ver registros
        xaxis_title='Fecha', #Nombre eje x
        yaxis_title='Venta (CLP)', #Nombre eje y
        template='plotly_white' #Diseño del grafico
    )

    # Eje Y muestra los valores completos sin notación científica
    figura_prediccion_validacion_venta.update_yaxes(
        tickformat="", #Mostrar numero real
        separatethousands=False, #Desactivar formato automatico
        exponentformat="none", #Desactivar notacion cientifica
        showexponent="none" #Desactivar exponente
    )

    st.plotly_chart(figura_prediccion_validacion_venta,use_container_width=True) #Visualizar en aplicacion


    #Visualizar metricas de error
    columna_mape_venta_validacion,columna_rmse_venta_validacion=st.columns(2) #Columnas

    columna_mape_venta_validacion.metric(
        "**Error Porcentual (MAPE):**",
        value=f"{mape_validacion_venta*100:.4f}%" #Valor
    )

    columna_rmse_venta_validacion.metric(
        "**Desviación en CLP (RMSE):**",
        value=f"${rmse_validacion_venta:,.0f}" #Valor
    )


    #-------------------------------------
    #Analisis prescriptivo
    #-------------------------------------
    st.subheader("**Acciones Basadas en la Tendencia de Venta**") #Titulo de la seccion

    #Renombrar columnas
    df_historico_recomendacion=df_venta_ivcm[['ds','y']].rename(columns={'y':'venta_historica'})
    df_prediccion_recomendacion=predecir_ventas[['ds', 'yhat']].rename(columns={'yhat':'venta_prediccion'})

    #Restar 1 año a cada fecha
    df_prediccion_recomendacion['ds_comportamiento']=df_prediccion_recomendacion['ds']-pd.DateOffset(years=1)

    #DataFrame de venta historica y predicha
    df_comportamiento_accion=df_prediccion_recomendacion.merge(
        df_historico_recomendacion,
        left_on='ds_comportamiento', #Fecha año anterior
        right_on='ds', #Fecha actual
        how='left',
        suffixes=('', '_historica')
    )

    #Calcular crecimiento
    df_comportamiento_accion['comportamiento']=((df_comportamiento_accion['venta_prediccion']-df_comportamiento_accion['venta_historica'])/df_comportamiento_accion['venta_historica'])
    df_comportamiento_accion['crecimiento_porcentaje']=df_comportamiento_accion['comportamiento']*100 #Porcentaje de crecimiento
    df_comportamiento_accion['crecimiento_porcentaje']=df_comportamiento_accion['crecimiento_porcentaje'].fillna(0) #Si no hay venta colocar cero

    #Agregar filtro
    fecha_inicio_crecimiento_accion,fecha_fin_crecimiento_accion=st.date_input('Seleccionar rango de fechas:',value=(df_comportamiento_accion['ds'].min(),df_comportamiento_accion['ds'].max())) #Rango de fechas
    df_filtrado_crecimiento=df_comportamiento_accion[(df_comportamiento_accion['ds']>=pd.to_datetime(fecha_inicio_crecimiento_accion))&(df_comportamiento_accion['ds']<=pd.to_datetime(fecha_fin_crecimiento_accion))] #Aplicar filtro

    #Grafico de comportamiento
    figura_comportamiento=px.line(
        df_filtrado_crecimiento,
        x='ds', #Fecha
        y='crecimiento_porcentaje', #Valores
        title='Crecimiento / Decrecimiento del Pronóstico (Fecha vs. fecha)', #Nombre
        labels={
            'ds':'Fecha', #Nombre eje x
            'crecimiento_porcentaje':'Decrecimiento / Crecimiento (%)' #Nombre eje y
        },
        color_discrete_sequence=['#1b88e7'] #Color
    )

    figura_comportamiento.add_hline(y=0, line_dash="dash") #Visualizacion crecimiento 

    st.plotly_chart(figura_comportamiento, use_container_width=True) #Ver en aplicacion


    #-------------------------------------
    #Tabla de comportamiento
    #-------------------------------------
    #Preparacion de DataFrame
    df_crecimiento=df_bc_crecimiento.merge(df_operacion,on='Fecha',how='left') #Unir datos
    df_crecimiento=df_crecimiento.drop(columns=['ipc', 'imacec']) #Eliminar IPC e IMACEC
    df_crecimiento.loc[df_crecimiento['Fecha'].isin(fechas_siempre_cero),'Venta_Retail']=0 #Aplicar cero a feriados irrenunciables
    df_crecimiento['Depto']=df_crecimiento['Depto'].fillna(0) #Reemplazar nulos por cero
    df_crecimiento['Depto']=df_crecimiento['Depto'].astype(int) #Transformar columna a entero


    #Divisiones
    divisiones={
    'Casa':[140,200,170,220,190,210,600],
    'Vestuario':[250,230,340,290,740,330,300,240,270,280],
    'Entretenimiento':[700,900,730,3000,100,160,110,790,530,120,180],
    'Hardlines':[500,870,150,770]
    }

    #Mapa de departamentos a division
    departamento_division = {departamento: division for division, departamentos in divisiones.items() for departamento in departamentos}

    #Considerar los ultimos 3 años de datos
    fecha_limite_comportamiento=df_crecimiento['Fecha'].max()-pd.Timedelta(days=1095)
    df_comportamiento_ultimos_3años=df_crecimiento[df_crecimiento['Fecha']>fecha_limite_comportamiento].copy()

    #Mapear cada departamento a la division
    df_comportamiento_ultimos_3años['division']=df_comportamiento_ultimos_3años['Depto'].map(departamento_division)

    #Agregar observaciones
    df_ventas_division=(
        df_comportamiento_ultimos_3años
        .groupby(['Fecha','division'],as_index=False)['Venta_Retail'].sum() #Agrupar por fecha y division, pero tambien sumar la venta
        .pivot(index='Fecha',columns='division',values='Venta_Retail') #Transformar filas en columnas
        .fillna(0) #Rellenar con cero el valor vacio  
        .reset_index() #Reiniciar el indice
    )

    #Eliminar duplicados
    df_comportamiento_ultimos_3años=df_comportamiento_ultimos_3años[['Fecha','Indice de Transacciones de Comercio Minorista']].drop_duplicates()
    #Combinar ventas por division con IVCM
    df_ventas_ivcm_ultimos_3años=df_ventas_division.merge(df_comportamiento_ultimos_3años,on='Fecha',how='left')

    #Extraer mes y año
    df_ventas_ivcm_ultimos_3años['mes']=df_ventas_ivcm_ultimos_3años['Fecha'].dt.month
    df_ventas_ivcm_ultimos_3años['año']=df_ventas_ivcm_ultimos_3años['Fecha'].dt.year


    #Obtener correlaciones
    meses=range(1,13) #Cantidad de meses
    correlaciones=[] #Lista

    for mes in meses: #Iterar por mes y obtener correlacion
        df_mes=df_ventas_ivcm_ultimos_3años[df_ventas_ivcm_ultimos_3años['mes']==mes]
        
        if len(df_mes)>1: #Calcular correlacion por division
            correlacion_casa=df_mes['Casa'].corr(df_mes['Indice de Transacciones de Comercio Minorista'],method='spearman')
            correlacion_vestuario=df_mes['Vestuario'].corr(df_mes['Indice de Transacciones de Comercio Minorista'],method='spearman')
            correlacion_entretenimiento=df_mes['Entretenimiento'].corr(df_mes['Indice de Transacciones de Comercio Minorista'],method='spearman')
            correlacion_hardlines=df_mes['Hardlines'].corr(df_mes['Indice de Transacciones de Comercio Minorista'],method='spearman')
            
            correlaciones.append({ #Guardar correlaciones
                'mes':mes,
                'Casa':correlacion_casa,
                'Vestuario':correlacion_vestuario,
                'Entretenimiento':correlacion_entretenimiento,
                'Hardlines':correlacion_hardlines
            })

    df_correlacion_division=pd.DataFrame(correlaciones) #Convertir a DataFrame

    #--------------------------
    #Matriz de recomendacion
    #--------------------------
    df_filtrado_crecimiento['mes']=df_filtrado_crecimiento['ds'].dt.month #Crear columna mes
    df_condicion_crecimiento=df_filtrado_crecimiento.merge(df_correlacion_division,on='mes',how='left') #Unir correlacion por mes
    df_condicion_crecimiento=df_condicion_crecimiento[['ds','crecimiento_porcentaje','Casa','Vestuario','Hardlines','Entretenimiento']] #Seleccionar columnas relevantes
    
    #Renombras columnas
    df_condicion_crecimiento=df_condicion_crecimiento.rename(columns={
        'crecimiento_porcentaje':'Crecimiento',
        'ds':'Fecha'
    })

    #-------------------------------------------------
    #Definir condiciones de tendencia de crecimiento
    #-------------------------------------------------
    condiciones=[
        df_condicion_crecimiento['Crecimiento']>5, 
        (df_condicion_crecimiento['Crecimiento']>=-5)&(df_condicion_crecimiento['Crecimiento']<=5),
        df_condicion_crecimiento['Crecimiento']<-5 
    ]

    #Definir los valores a asignar
    valores=['Crecimiento (> 5%)','Estable (-5% a 5%)','Caida (< -5%)']

    #Crear columna de tendencia
    df_condicion_crecimiento['Tendencia']=np.select(condiciones,valores,default='NaN')

    #-------------------------------------
    #Clasificar sensibilidad
    #-------------------------------------
    divisiones_clasificar=['Casa','Vestuario','Hardlines','Entretenimiento']

    for division in divisiones_clasificar:
        condiciones = [
            df_condicion_crecimiento[division]>0.5,                            
            df_condicion_crecimiento[division]<0.3,                            
            (df_condicion_crecimiento[division]>=0.3)&(df_condicion_crecimiento[division]<=0.5)  
        ]
        categoria=['Perceptivo al Ciclo','Independiente','Continua con la Tendencia']
        
        #Crear columna sensibilidad
        df_condicion_crecimiento['Sensibilidad de '+division]=np.select(condiciones,categoria,default='NaN')

    #------------------------
    #Impacto
    #------------------------
    def clasificar_impacto(tendencia,sensibilidad):
        '''clasificar impacto operativo en base a la tendencia y la sensibilidad'''
        if tendencia=='Crecimiento (> 5%)':
            if sensibilidad=='Perceptivo al Ciclo':
                return'ALTO'
            elif sensibilidad in ['Independiente','Continua con la Tendencia']:
                return'MEDIO'

        elif tendencia=='Estable (-5% a 5%)':
            return'BAJO'

        elif tendencia=='Caida (< -5%)':
            if sensibilidad=='Perceptivo al Ciclo':
                return'ALTO'
            elif sensibilidad in ['Independiente','Continua con la Tendencia']:
                return'MEDIO'

        return'NaN'

    #Aplicar funcion clasificar impacto
    for division in divisiones_clasificar: #Iterar por cada division para obtener el impacto
        df_condicion_crecimiento[f'Impacto Operativo de {division}']=df_condicion_crecimiento.apply( #Crear columna
            lambda row:clasificar_impacto(
                row['Tendencia'],
                row[f'Sensibilidad de {division}']
            ),
            axis=1 #Aplicar funcion
        )

    #-------------------------------
    #Accion
    #-------------------------------
    def determinar_accion(tendencia,perfil,impacto):
        """determinar la accion segun la tendencia,sensibilidad e impacto"""
        if tendencia=='Crecimiento (> 5%)' and perfil=='Perceptivo al Ciclo' and impacto=='ALTO':
            return'Abastecimiento'
        
        elif tendencia=='Crecimiento (> 5%)' and perfil=='Continua con la Tendencia' and impacto=='MEDIO':
            return'Abastecimiento'
        
        elif tendencia=='Crecimiento (> 5%)' and perfil=='Independiente' and impacto=='MEDIO':
            return'Monitorear actividad comercial'
        
        elif tendencia=='Estable (-5% a 5%)' and perfil=='Perceptivo al Ciclo' and impacto=='BAJO':
            return'Monitorear actividad comercial'
        
        elif tendencia=='Estable (-5% a 5%)' and perfil=='Continua con la Tendencia' and impacto=='BAJO':
            return'Mantener actividad comercial'
        
        elif tendencia=='Estable (-5% a 5%)' and perfil=='Independiente' and impacto=='BAJO':
            return'Mantener actividad comercial'   
        
        elif tendencia=='Caida (< -5%)' and perfil=='Perceptivo al Ciclo' and impacto=='ALTO':
            return'Activar actividad comercial'
        
        elif tendencia=='Caida (< -5%)' and perfil=='Continua con la Tendencia' and impacto=='ALTO':
            return'Reducir pedidos'
        
        elif tendencia=='Caida (< -5%)' and perfil=='Independiente' and impacto=='MEDIO':
            return'Ajustar gastos'   
        else:
            return'NaN'

    #Aplicar funcion determinar accion
    for division in divisiones_clasificar: #Iterar por cada division para obtener la accion
        df_condicion_crecimiento[f'Accion de {division}']=df_condicion_crecimiento.apply( #Crear columna
            lambda row:determinar_accion(
                row['Tendencia'],
                row[f'Sensibilidad de {division}'],
                row[f'Impacto Operativo de {division}']
            ),
            axis=1 #Aplicar funcion
        )

    #Eliminar columnas no relevantes
    df_accion_recomendada=df_condicion_crecimiento.drop(columns=['Crecimiento', 'Casa', 'Vestuario', 'Hardlines', 'Entretenimiento'])

    st.dataframe(df_accion_recomendada) #Visualizar tabla en aplicacion


