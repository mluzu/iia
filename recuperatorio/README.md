1 - El modelo de regresión lineal aproxima la función que relaciona las variables de entrada con las de salida, 
asumiendo que esta es lineal (bias). La variable de salida debe ser continua (variable cuantitativa). La aproximación
más común implica minimizar utiliza least squares para estimar los coeficientes. 

Una forma de obtener la fórmula cerrada para los coeficientes es a partir la RSS (residual sum of squares). Esta 
expresión es la suma de los cuadrados de las diferencias entre cada valor verdadero de la variable de salida y su 
estimación. Luego se diferencia respecto a los coeficientes y se iguala a cero para finalmente despejar. Los coeficientes
obtenidos permiten minimizar MSE. 
Este resultado asume, aparte de la linealidad, que las variables de entrada están descorrelacionadas entre sí y que
tienen varianza constante. Por otra parte asume que la distribución de la salida es gaussiana.

Desde un punto de vista estadístico, la función con la que aproximamos la relación subyacente del conjunto de datos, 
es decir la función de regresión, no es otra cosa que la función de distribución del valor esperado de la variable de 
salida dadas las variables de entrada: f(x) = E(Y|X=x). A partir de esto podemos profundizar en el enfoque bayesiano



Para obtener la fórmula cerrada 
2 - Exploramos el dataset para entender su constitución. Levantamos los datos en un array estructurado de numpy, 
descartando la primer columna (índice) y la primer fila (encabezado). Tenemos 1000 muestras y 10 features, más la 
variable de salida. No hay faltantes en ninguna muestra. La salida es una variable binaria y está balanceada. 
El dataset se particiona con 80% para entrenamiento y 20% para testeo.
En todos los casos se utilizó la métrica acurracy para seleccionar el mejor modelo, es decir el número de predicciones 
correctas sobre el total de predicciones.
Se usa k-folds con k=5 para ajuste de los hiperparámetros bias y learning rate. Se loguea loss cada 100 epochs.
Obtuvimos el mejor modelo con bias y learning rate 0.001.

3 - Se entrena un modelo de regresión logística con regularización Ridge como función de costo.
Este se obtiene agregando un segundo término a la función basada en least squares. Este término se conoce como shrinkage 
penalty y tiene como efecto que los coeficientes que minimizan la expresión se sean pequenos, tendiendo a cero a medida 
que el valor de lambda crece. Básicamente restringe al norma del vector de parámetros. 
La ventaja de usar este método se explica por el trade-off entre varianza y bias. Lambda hace más rígido 
el modelo a medida que crece, con el consecuente incremento de la varianza y reducción del bias. El resultado debería 
ser un mejor desempeño del modelo en el set de testeo porque el modelo gana capacidad de generalizar.

Por otra parte, un efecto de la regularización Ridge es que las variables de entrada que tienen menos influencia sobre 
la salida tengan asociados coeficientes más pequeños. Esta propiedad lo hace interesante para compararlo con un método
de reducción de dimensiones como PCA.