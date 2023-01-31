# DA-promo-c-modulo-3-sprint-1-Sila-Silvia

En este repositorio se encuentran los ejercicios realizados en el pair programming modulo 3 - Sprint 1 por Silvia Gordón y Sila Rivas. 

Dicho repositorio se divide en dos carpetas:

.- Carpeta Regresión Lineal: Constituye a los ejercicios realizados en referencia a los temas de Regresión Lineal.
    
    - Leccion-01-Intro-Machine-Learning.

    - Leccion-02-Test-Estadisticos.

    - Leccion-03-Covarianza-Correlacion.

    - Leccion-04-Asunciones.

    - Leccion-05-Normalización.

    - Leccion-06-Estandarización.

    - Leccion-07-ANOVA.

    - Leccion-08-Encoding.

    - Leccion-09-Regresion-Lineal-Intro.

    - Leccion-10-Regresion-Lineal-Metricas.

    - Leccion-11-Regresion-Lineal-Decision-Tree.

    - Leccion-12-Regresion-Lineal-Random-Forest.

.- Carpeta Regresión Logística: Constituye a los ejercicios realizados en referencia a los temas de Regresión Logística.
    
    - Leccion-13-Regresion-Logistica-EDA.

    - Leccion-14-Regresion-Logistica-Preprocesado.

    - Leccion-15-Regresion-Logistica-Intro.

    - Leccion-16-Regresion-Logistica-Metricas.

    - Leccion-17-Regresion-Logistica-Decision-Tree.

    - Leccion-18-Regresion-Logistica-Random-Forest.

Las librerías utilizadas en este repositorio han sido:

# Tratamiento de datos
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm

# Gráficos
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Modelado y evaluación
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , cohen_kappa_score, roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#  Crossvalidation
# ------------------------------------------------------------------------------
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics

# Estandarización variables numéricas y Codificación variables categóricas
# ------------------------------------------------------------------------------
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder # para realizar el Label Encoding 
from sklearn.preprocessing import OneHotEncoder  # para realizar el One-Hot Encoding

# Estadísticos
# ------------------------------------------------------------------------------
import statsmodels.api as sm
from statsmodels.formula.api import ols
import researchpy as rp
from scipy.stats import skew
from scipy.stats import kurtosistest

# Gestión datos desbalanceados
# ------------------------------------------------------------------------------
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek

# Configuración warnings
# ------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

# Establecer tamaño gráficas
# ------------------------------------------------------------------------------
plt.rcParams["figure.figsize"] = (15,15)