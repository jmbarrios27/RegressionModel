import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})
import joblib

# MACHINE LEARNING
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ALGORITMOS Y METRICAS DE EVALUACIÓN.
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# DATA VISUALIZATION
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Reading Data
dataframe = pd.read_csv('data.csv', sep=',')
print(dataframe)

# Data split
X = dataframe.iloc[:, :2]
y = dataframe.iloc[:, -1]


# Function to run Linear regression Algorithm.
def modeling_linear_regression(X_features, y_features):
    # Data partition y Modelo.
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_features, train_size= 0.80, random_state=100)
    
    # Shape
    print('Data Shape')
    print('X_train', X_train.shape)
    print('X_test', X_test.shape)
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)
    print()
    
    # Creando Modelo de Regressión Lineal.
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train, y_train)
    
    # Creando Predicciones
    y_pred_linear = linear_regression_model.predict(X_test)
    
    # Creando métrica de RMSE
    mse_linear = r2_score(y_test, y_pred_linear)
    rmse_linear = np.sqrt(mse_linear)
    
    # Evaluando Modelo
    print('METRICAS DE EVALUACIÓN')
    print('METRICA R2: {:.2f}'.format(r2_score(y_test, y_pred_linear)))
    print('RMSE: {:.2f}'.format(rmse_linear))
    print('MAE: {:.2f}'.format(median_absolute_error(y_test, y_pred_linear)))
    
    return linear_regression_model, y_pred_linear


    # Calling Function.
# Llamando a la función y modelando.
modelo, predicciones = modeling_linear_regression(X_features=X, y_features=y)