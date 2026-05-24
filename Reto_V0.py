""" Reto: Predicción de cancelaciones de hotel.
Los datos de cancelaciones de reservaciones en un hotel, tiene un fuerte desbalance.
Hay mucho menos cancelaciones, que cancelaciones, lo cual tiene sentido, de otra manera mejor cerramos el hotel.
El desbalance es medianamente grande que sí no se aplican medidas para tratar este problema, no importa que tan 
bueno sea tu algoritmo muy probablemente no va a lograr detectar los casos en los que sí se van a cancelar las
reservaciones, que es la clase que nos interesa. Nos interesa porque como hotel, lo que quisieramos es evitar
tener cuartos vacios a lo largo del año, es preferible invertir un poco más de dinero para influir en los clientes
que quieren cancelar para que no cancelen. Puedes implementar alguna camapaña para promover o emocionar a los
clientes para que vengan, pero primero es necesario identificar de manera sistemática quiénes serán. No quieres
enviarle a todo mundo una camapaña porque te cuesta dinero (supón que les regalas una noche extra o una cena, etc).
En este cuaderno, les muestro la manera en la que pueden aplicar downsampling, que consiste en igualar el número
de observaciones de la clase mayoritaria al número de observaciones de la clase minoritaria. Para hacer esto hay
que sacar muestras aleatorias sin reemplazo. Con esto vamos a lograr un conjunto balanceado en el que nuestros
algorítmos se desempeñaran mejor.

Setup del ambiente de trabajo en Google Colab.

* Para que puedas reproducir los códigos de la clase es importante que fijes las librerías de Python en tu cuaderno
de Colab a aquellas que se utilizaron en clase.
* Recuerda que el software evoluciona todo el tiempo y que si no fijamos las librerías de nuestro proyecto corremos
el riesgo de que se rompa el código en el futuro.
* El Google Colab instala por lo general las últimas versiones de las librerías.

"""

# Liberías estandar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Funciones de Scikit-Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, 
    train_test_split
)

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix
)

# Funciones de Yellow Brick
from yellowbrick.target import ClassBalance
from yellowbrick.model_selection import (ValidationCurve, FeatureImportances)
from yellowbrick.classifier.threshold import DiscriminationThreshold
from yellowbrick.classifier import (
    ConfusionMatrix, 
    ClassPredictionError, 
    ClassificationReport,
    PrecisionRecallCurve, 
    ROCAUC, 
    ClassPredictionError
)

# Importar xgboost
import xgboost as xgb

# Configurar visualizaciones
sns.set_theme(style="whitegrid")
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

"""
Cargar los datos

Cargamos los datos. Nuestra variable a predecir es `is_canceled`. (1) es cancelada y (0) no cancelada.
"""

# https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11
tbl_data = (
    # Read data from the internet
    pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv")
)

tbl_data

pd.crosstab(tbl_data.is_canceled, tbl_data.reservation_status)

"""
Análisis Exploratorio de Datos (EDA)
* Aqui hay un análisis exploratorio de datos para entender los datos.
* Revísalo cuidadosamente para entender que tienen los datos.
Este reporte se construye con una librería que se llama `pandas_profiling`. Automatiza muchas
gráficas y tablas resumen que se usan en los EDA. es útil como herramienta, pero en la práctica
no es suficiente, hay que seguir explorando los datos con mayor detalle. Para este proyecto es
suficiente y un buen punto de partida. Si quieres conocer más puedes checar la
[documentación](https://pypi.org/project/pandas-profiling/).

# En Google Colab ejecuta esta celda una sola vez. Después de instalar y reiniciar
# la sesión puedes comentar esta linea.
"""

from ydata_profiling import ProfileReport
profile=ProfileReport(tbl_data, title="Pandas Profiling Report")
profile

# Checar el desbalance en los datos.
(tbl_data
     .groupby(['is_canceled'])
     .size()
     .reset_index(name = 'n_reservaciones')
     .assign(pct = lambda df_: df_.n_reservaciones / df_.n_reservaciones.sum() * 100)
     .round(1)
)

fig, ax = plt.subplots(figsize = (8, 6))
sns.barplot(
    data = (tbl_data
            .groupby(['is_canceled'])
            .size()
            .reset_index(name = 'n_customers')),
    x = 'is_canceled',
    y = 'n_customers'
)
plt.title("Does customer check-in with children?")
plt.plot()

"""
Corregir desbalance con downsampling

* Construimos una función para hacer el downsampling, lo único que necesitas es pandas.
* Probablemente, no conozcas lo que hace `query`, es una función que encadenas a un pandas
dataframe y te permite filtrar las filas
(checa la [documentación](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)).
* La función `sample` te permite encadenarla a un pandas dataframe y sacar filas de manera
aleatoria (checa la [documentación](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html)).
"""

def fun_downsample(tbl):
    '''
    Función para hacer downsampling
    tbl: Son los datos originales en forma de dataframe.
    '''
    # Filtramos la clase mayoritaria y sacamos de manera aleatoria N observaciones.
    # N es igual al número de observaciones en la clase minoritaria.
    # Observa que fijo una semilla en el random_state para garantizar reproducibilidad.
    tbl_reservations_not_cancelled = (
        tbl
            .query('is_canceled == 0')
            .sample(
                n = tbl.groupby(['is_canceled']).size()[1], 
                random_state=42)
    )
    # Filtramos la clase minoritaria.
    tbl_reservations_cancelled = tbl.query('is_canceled == 1')
    
    return pd.concat([
        tbl_reservations_not_cancelled,
        tbl_reservations_cancelled
    ])

tbl_downsampled_data = fun_downsample(tbl_data)

"""
* Observa como igualamos las proporciones de las clases a 50:50
* Desde luego que perdemos mucha información, sin embargo, el costo no es superior al beneficio.
* El beneficio será que nuestro algorítmo será capaz de atrapar más verdaderos positivos que sin debalance.
* Puede haber casos en los que esto no funcioné y tengamos que aplicar otra distribución, pero por ahora lo dejamos así.
"""

(tbl_downsampled_data
     .groupby(['is_canceled'])
     .size()
     .reset_index(name = 'n_reservaciones')
     .assign(pct = lambda df_: df_.n_reservaciones / df_.n_reservaciones.sum() * 100)
     .round(1)
)

fig, ax = plt.subplots(figsize = (8, 6))
sns.barplot(
    data = (tbl_downsampled_data
            .groupby(['is_canceled'])
            .size()
            .reset_index(name = 'n_customers')
            .assign(is_canceled = lambda df_: df_.is_canceled.replace({0: 'not cancelled', 1: 'cancelled'}))),
    x = 'is_canceled',
    y = 'n_customers'
)
plt.title("Distribución de reservaciones con cancelaciones")
plt.show()

"""
Construimos los conjuntos de entrenamiento

* Observa que para construir los conjuntos de entrenamiento y prueba usamos `tbl_downsampled_data`
los nuevos datos y no `tbl_data` que son los datos originales.
"""

from sklearn.model_selection import train_test_split

y = tbl_downsampled_data.is_canceled
X = tbl_downsampled_data.drop(columns = 'is_canceled')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

"""
A partir de aqui puedes seguir el flujo de ML que hemos visto en clase ...

* Dado que lo que queremos es resolver un problema de clasificar cada reservación como cancelada y no cancelada,
lo que queremos construir es un modelo de aprendizaje supervisado para clasificación.
* Aqui tienes que ejecutar el código visto en la clase asíncrona del Profundiza.

Paso 1. Selecciona y extrae las características de tu modelo.

* Recuerda que tienes que aplicar tus transfomaciones y selección de variables tanto al conjunto de entrenamiento
como al conjunto de prueba. Tu elige como hacerlo.

* Con lo único que tienes que tener cuidado es con las variables categóricas que tienen muchas clases, porque
puede ser que al momento de convertirlas en dummies no tengas el mismo número de columnas o tengas columnas
diferentes. TIP: No uses `pd.get_dummies()` utiliza `OneHotEncoder` de `scikit-learn`. 

* TIP: Toma el código que vimos en la segunda clase, que ya implementa el OneHotEncoder y modifica, lo que tengas que modificar.

* TIP: No selecciones la variable de `reservation_status`, porque si observas en el EDA, es lo mismo que la variable
`is_cancelled`. No entrenas un modelo pasándole la respuesta, obvio va a sacar score perfectp. Por eso es muy
importante que revises el análisis exploratorio de los datos y no metas todo a la licuadora.
"""

X_test

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

variable_selection_numeric = ['OverallQual', 'GrLivArea', 'YearBuilt', '1stFlrSF']
variable_selection_categoric = ['Neighborhood','MSZoning']               # <--- Selecciona las variable categóricas

def prep_for_ml(tbl_train, tbl_test):
    '''Clean X_train and X_test 
        1) Select continuous and categorical variables.
        2) Convert categorical variables to one hot encoding
        3) Concatenate clean dataframes
    '''
    # Continuous variables
    tbl_num_train = tbl_train.loc[:, variable_selection_numeric]  
    tbl_num_test = tbl_test.loc[:, variable_selection_numeric]  
    
    scaler = StandardScaler()
    scaler.fit(tbl_num_train)
    X_train_standarized = scaler.transform(tbl_num_train)
    X_test_standarized = scaler.transform(tbl_num_test)
    
    
    #############################################################
    
    # Categorical variables
    tbl_cat_train = tbl_train.loc[:, variable_selection_categoric]
    tbl_cat_test = tbl_test.loc[:, variable_selection_categoric]
    
    ohe = OneHotEncoder(drop = 'first', sparse = False)
    ohe.fit(tbl_cat_train)
    col_names = ohe.get_feature_names_out()

    tbl_ohe_cat_train = pd.DataFrame(
        ohe.transform( tbl_cat_train )
    )
    
    tbl_ohe_cat_test = pd.DataFrame(
        ohe.transform( tbl_cat_test )
    )
    # Add new column names
    tbl_ohe_cat_train.columns = col_names
    tbl_ohe_cat_test.columns = col_names

    #############################################################
    
    # Join transformed continuous + categorical variables
    tbl_train_clean = pd.concat([X_train_standarized.reset_index(drop = True), tbl_ohe_cat_train], axis = 1)    
    tbl_test_clean = pd.concat([X_test_standarized.reset_index(drop = True), tbl_ohe_cat_test], axis = 1)
    
    return (tbl_train_clean, tbl_test_clean)

X_train_clean, X_test_clean = prep_for_ml(X_train, X_test)

"""
* Estas líneas nos sirven para validar que las columnas de los tres conjuntos de datos después de aplicar
el preprocesamiento están correctas.
* Si genera error es porque no tienes las mismas columnas, puedes investigar por qué?
* HINT: Los errores en el código son comúnes, y están diseñados para decirte que es lo que está fallando.
Cómo cientifico de datos, pasas una parte importante debuggeando código. Si tienes un error que no entiendes,
lo copias y pegas en Google y puedes investigar que dice la comunidad y cómo se resuelve. O puedes preguntar.
"""

assert X_train_clean.columns.tolist() == X_test_clean.columns.tolist()

"""
Paso 2: Construye un modelo base

* Construye un modelo sencillo. Recuerda que estas viendo problemas de clasificación, por lo que tienes que escoger la implementación correcta. Por ejemplo:
    - [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
    - [Support Vector Classification (SVC¶)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
    - [K nearest neighbors (KNeighborsClassifier¶)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
    - [Decision Trees (DecisionTreeClassifier)](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
    - [Random Forest (RandomForestClassifier)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
    - [Xgboost (XGBClassifier)](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
    
* Entrena el modelo y calcula la métricas de error para un problema de clasificación: precision, recall, f1 score,
y la matriz de confusión. Utiliza las funciones de yellowbrick que vimos en clase.

* Analiza los resultados, que implica para el hotel. ¿Qué dicen los Falsos Negativos, Falsos Positivos? ¿Qué implica
una precision o recall de X en la operación del hotel utilizando tu modelo?
"""

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y)

from yellowbrick.classifier import ConfusionMatrix
fig, ax = plt.subplots(figsize = (5,5))
# Creamos la matrix de confusión
cm = ConfusionMatrix(
    best_model, 				# Pasar el estimador del mejor modelo que 
    							# se obtiene de la validación cruzada. 
    classes=['no', 'yes']		# Indicar las etiquetas de las clases.
    							# Cuidado con el orden de los nombres.
)
# Ajustamos la visualización a los datos de entrenamiento
cm.fit(X_train, y_train)
# Evaluamos los errores en la predicción utilizando los datos de validación.
cm.score(X_val, y_val)

from yellowbrick.classifier import ClassificationReport
fig, ax = plt.subplots(figsize = (6,5))
visualizer = ClassificationReport(
    best_model,					# Pasar el estimador del mejor modelo que 
    							# se obtiene de la validación cruzada. 
    classes=['no', 'yes']		# Indicar las etiquetas de las clases.
    							# Cuidado con el orden de los nombres.
)
# Ajustamos la visualización a los datos de entrenamiento
visualizer.fit(X_train, y_train)
# Evaluamos los errores en la predicción utilizando los datos de validación.
visualizer.score(X_val, y_val)
# Dibjuamos la visualización.
visualizer.show();

"""
Paso 3: Busca los mejores hiperparámetros de tu modelo

* Para esto puedes utilizar el `GridSearch` o el `RandomSearch`, tu elige la implementación que quiras,
solo ten cuidado con los parámetros que les tienes que pasar porque hay sutilezas. Tienes los códigos
de las clases y la documentación como referencia.

* Escoge los hiperparámetros correctos. TIP: Puedes tomar como referencia la tabla del libro que les
pasé por correo de Canvas. Puedes ajustar los rangos de los hiperparámetros según los resultados que observes.

* Calcula los las mismas métricas que sacaste en el modelo base, ¿mejoraron los resultados?
"""

params = {
    'n_estimators':[150],
    'learning_rate':[.01],
    'max_depth':[10],
    'subsample':[0.75],
    'colsample_bytree':[0.75],
    'colsample_bylevel':[0.75],
    'reg_lambda':[0.01]
}

grid_search = RandomizedSearchCV(
	clf_xgb,					# Especificar el modelo (estimador) 
	params,						# Especificar los parámetros de la malla 
	scoring = 'roc_auc', 		# Especificar la métrica de evaluación
	cv = 10,					# Especificar los k-cortes de la validación cruzada 
	n_iter=2,					# Especificar número de modelos a explorar aleatoriamente 
    return_train_score=True,# Agregar el error de entrenamiento
	n_jobs=-1					# Especificar el número de CPUs para paralelizar el trabajo
    							# -1= todos los disponibles
)

# Paso 4: Prueba con otro modelo y ajusta sus hiperparámetros como en el paso 3.

"""
Paso 5: Selecciona el mejor modelo y calcula las predicciones

* Utiliza el mejor modelo para calcular las predicciones utiliznado el conjunto de prueba.
    - Predice la clase
    - Predice la probabilidad de cancelación
* Grafica la curva ROC del mejor modelo.
* Reporta la precision y el recall.
* Interpreta los resultados en función de lo que estos implican pra el modelo si utilizan tu modelo.
"""

"""
Paso 5: Genera un CSV con tus predicciones para compartir con el hotel, utilizando tu mejor modelo entrenado con los datos de prueba.

* Reporta en tu CSV la predicción de la clase y la predicción de la probabilidad de cancelación.
"""