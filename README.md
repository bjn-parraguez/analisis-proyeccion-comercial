Pronóstico diario de ventas agregadas

Descripción:
Este proyecto, diseñado para ejecutarse en entorno local, desarrolla un modelo de pronóstico de ventas agregadas a horizonte anual con el objetivo de fundamentar el proceso de planificación del área comercial.
La solución responde a la ausencia de un sistema automatizado de proyección en la organización y aborda directamente la alta dependencia histórica de estimaciones subjetivas, reemplazándolas por un enfoque analítico, cuantitativo y reproducible.


Objetivo:
Obtener un proceso de pronóstico de ventas diarias fidedigno que permita optimizar la planificación y así apoyar la toma de decisiones.

Metodología:

- Integración de datos internos y externos de la organización.
  
- Análisis del comportamiento del consumidor y oportunidad de mercado a partir de información histórica y las proyecciones generadas. 

- Modelado predictivo con Prophet.
  
- Análisis prescriptivo para decisiones estratégicas en torno al crecimiento. 

- Validación a través de ventana rodante de 60 días, mediante métricas de error como MAPE y RMSE. 

Resultados:
En un entorno operativo el modelo se evaluó a través de una ventana rodante, demostrando un desempeño aproximado de 12,9% (MAPE), lo cual es consistente en torno a la escala numérica de la serie temporal.

Estructura del repositorio:

- “retail_forecast_app.py”: Integra la extracción de datos de ventas, el análisis del comportamiento del consumidor y oportunidades de mercado a través del enfoque descriptivo, predictivo y prescriptivo. Además, incluye la aplicación local interactiva desarrollada para el proyecto.
  
- “ingesta_datos_api_banco_central_de_chile.py”: Extracción de datos económicos desde la API del Banco Central de Chile.

Nota de Arquitectura y confidencialidad:
El repositorio contiene únicamente la etapa de modelamiento, aplicación e integración funcional. Las etapas previas de limpieza, análisis exploratorio de datos (EDA), selección de variables y prueba de supuestos estadísticos se desarrollaron en entornos de experimentación durante la fase de investigación y desarrollo. 
Por motivos de seguridad, modularidad y buenas prácticas, la arquitectura se presenta desacoplada en dos componentes independientes (ingesta de API externa y pipeline predictivo/interfaz). Asimismo, las credenciales y datos corporativos sensibles han sido omitidos. 

Herramientas:

- Desarrollo del análisis - Python
  
- Extracción de datos - SQL y API

- Visualización y aplicación local - Streamlit 

Autor:
Benjamín Parraguez |
Proyecto académico 
