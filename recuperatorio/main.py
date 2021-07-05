from dataset import Dataset, PCA, split
from models import LogisticRegression
import numpy as np
from kfolds import lg_k_folds
from metrics import Accuracy, Recall, Precision

# 2 - Pre-procesamiento del dataset

"""
Exploramos el dataset para entender su constitución. Levantamos los datos en un array estructurado de numpy, 
descartando la primer columna (índice) y la primer fila (encabezado). Tenemos 1000 muestras y 10 features, más la 
variable de salida. No hay faltantes en ninguna muestra. La salida es una variable binaria y está balanceada. 
El dataset se particiona con 80% para entrenamiento y 20% para testeo.
En todos los casos se utilizó la métrica acurracy para seleccionar el mejor modelo, es decir el número de predicciones 
correctas sobre el total de predicciones.
"""

# carga e inspección del dataset
data = Dataset('recuperatorio_dataset.csv')
# missing = len(data.find_missing())
# print(f'El número de faltantes en el dataset es {missing}')
print(f'El dataset tiene {data.muestras()} muestras y {data.features() - 1} features')


# split dataset
X_train, X_test, y_train, y_test = data.split(0.8)
print(f'Formato del dataset de entrenamiento: {X_train.shape}')
print(f'Formato de etiquetas : {y_train.shape}')

# variable de salida
salida = np.unique(y_train)
balance = np.unique(y_train,  return_counts=True)
print(f'La salida tiene {len(salida)} categorías con valor {salida[0]} y {salida[1]}')
print(f'La categoría {balance[0][0]} está representada por el {(balance[1][0]/len(y_train))*100} de las muestras')
print(f'La categoría {balance[0][1]} está representada por el {(balance[1][1]/len(y_train))*100} de las muestras')
print('*************\n\n')

# 3 - Regresión Logística con mini-batch
"""
Se usa k-folds con k=5 para ajuste de los hiperparámetros bias y learning rate. Se loguea loss cada 100 epochs.
Obtuvimos el mejor modelo con bias y learning rate 0.001  
"""


def get_best_metric(kfolds_result, hiper, metric='acurracy'):
    acc = np.zeros(len(kfolds_result))
    for i, kfold in enumerate(kfolds_result):
        acc[i] = kfold.get(metric)
    idx = np.argmax(acc)
    metric_best = kfolds_result[idx]
    best_hiper = hiper[idx]
    return best_hiper, metric_best

# 3.a - Selección de modelo por hiperparámetro: bias
lr = 0.1
b = 16
epochs = 1000
verbose = False

print('HIPERPARMETRO: bias')
kfolds = []
bias_list = [True, False]
for bias in bias_list:
    result = lg_k_folds(X_train, y_train.reshape(-1, 1), lr, b, epochs, lamda=None, bias=bias, k=5, verbose=verbose)
    kfolds.append(result)

best_bias, metric = get_best_metric(kfolds, bias_list)

msg = 'Modelo con bias' if best_bias else 'Modelo sin bias'
acc = metric['accuracy']
rec = metric['recall']
pre = metric['precision']
print(f'\n\n{msg}: accuracy={acc}\trecall={rec}\tprecision={pre}')
print('*************\n\n')

# 3.b - Selección de modelo por hiperparámetro: learning rate

print('HIPERPARAMETRO: Learning rate')
kfolds = []
lr_list = np.linspace(0.001, 0.01, 10)
for lr in np.linspace(0.001, 0.01, 10):
    result = lg_k_folds(X_train, y_train.reshape(-1, 1), lr, b, epochs, lamda=None, bias=best_bias, k=5, verbose=verbose)
    kfolds.append(result)

best_lr, metric = get_best_metric(kfolds, lr_list)
acc = metric['accuracy']
rec = metric['recall']
pre = metric['precision']
print(f'\n\nLearning rate {best_lr}: accuracy={acc}\trecall={rec}\tprecision={pre}')
print('*************\n\n')

# 4 - Regresión Logística con mini-batch y regularización ridge

# 4.a - Fit del modelo obtenido

print("MEJOR MODELO OBTENIDO (Least Square)")
print(f'Hiperparametros: bias: {best_bias} \t Learning Rate {best_lr} \t')
logistic_regression = LogisticRegression(best_bias)
logistic_regression.fit(X_train, y_train.reshape(-1, 1), best_lr, b, epochs, None)
predictions = logistic_regression.predict(X_test)
metrics = [Accuracy(), Precision(), Recall()]
results = {}
for metric in metrics:
    name = metric.__class__.__name__
    results[name] = metric(y_test, predictions[:, 0])
    print('{metric}: {value}'.format(metric=name, value=results[name]))
print('*************\n\n')


"""
Se una entrena un modelo de regresión logística con regularización Ridge como función de costo.
Se agrega un segundo término a la función basada en least squares.  Este término  se conoce como shrinkage penalty y 
tiene como efecto que los coeficientes que minimizan la expresión se sean pequenos, tendiendo a cero a medida que el 
valor de lambda crece. Básicamente restringe al norma del vector de parámetros. 
La ventaja de usar este método se explica por el trade-off entre varianza y bias. Lambda hace más rígido 
el modelo a medida que crece, con el consecuente incremento de la varianza y reducción del bias. El resultado debería 
ser un mejor desempeño del modelo en el set de testeo porque el modelo gana capacidad de generalizar.

Por otra parte, un efecto de la regularización Ridge es que las variables de entrada que tienen menos influencia sobre 
la salida tengan asociados coeficientes más pequeños. Esta propiedad lo hace interesante para compararlo con un método
de reducción de dimensiones como PCA.
"""

lr = 0.1

print('HIPERPARAMETRO: lamda')
kfolds = []
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
for lamda in alpha_ridge:
    result = lg_k_folds(X_train, y_train.reshape(-1, 1), best_lr, b, epochs, lamda=lamda, bias=best_bias, k=5, verbose=verbose)
    kfolds.append(result)

best_lamda, metric = get_best_metric(kfolds, alpha_ridge)
acc = metric['accuracy']
rec = metric['recall']
pre = metric['precision']
print(f'\n\nLambda {best_lamda}: accuracy={acc}\trecall={rec}\tprecision={pre}')
print('*************\n\n')

# 5 - PCA

"""
Con el mejor modelo obtenido en los puntos anteriores se hace fit con una entrada formada por dos componentes
principales.
"""

# 5.a - Aplicación de PCA con dos componentes principales
print('Reducción de dimensiones con PCA')
X, y = data.array_X_y()
pca = PCA(X)
pca_X = pca.fit(2)
pca.plot()
X_train, X_test, y_train, y_test = split(pca_X, y, 0.8)
print(f'Formato del dataset de entrenamiento: {X_train.shape}')
print(f'Formato de etiquetas : {y_train.shape}')

print("\n\nFit del modelo con dos componentes principales")
logistic_regression = LogisticRegression(best_bias)
logistic_regression.fit(X_train, y_train.reshape(-1, 1), best_lr, b, epochs, best_lamda)
predictions = logistic_regression.predict(X_test)
for metric in metrics:
    name = metric.__class__.__name__
    results[name] = metric(y_test, predictions[:, 0])
    print('{metric}: {value}'.format(metric=name, value=results[name]))
print('*************\n\n')
print('FINALIZADO')
