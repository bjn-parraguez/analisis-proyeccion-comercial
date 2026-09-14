#-----------------------------------------------
#Librerias
#-----------------------------------------------
import bcchapi #Banco Central de Chile
from datetime import datetime

#-----------------------------
#Conectar con BCCH
#-----------------------------
siete=bcchapi.Siete("correo de usuario","contraseña de usuario")

#-----------------------------
#Descargar datos
#-----------------------------
df=siete.cuadro(
    series=["F034.VDCMSES.IND.DBC.2018.0.D","F074.IPC.VAR.Z.Z.C.M","F032.IMC.IND.Z.Z.EP18.N03.Z.1.M"],  #Codigo de la serie de datos
    nombres=["Indice de Transacciones de Comercio Minorista","ipc","imacec"], #Nombre de la serie de datos                    
    desde="2022-01-01", #Fecha de inicio 
    hasta=datetime.today().strftime("%Y-%m-%d") #Fecha final
)

#Renombrar columna de fecha
df=df.reset_index().rename(columns={"index":"Fecha"})

#-----------------------------
#Guardar en formato CSV
#-----------------------------
df.to_csv("df_BCCH.csv", index=False, encoding="utf-8-sig")
