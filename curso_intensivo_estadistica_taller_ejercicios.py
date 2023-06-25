print("*****************************")
print("TALLER EJERCICIOS ESTADÍSTICA")
print("*****************************")

# Ejercicios para consolidar
# Todo aplicado a Data Science. Antes hay que instalar:

# conda install pandas
# conda install scipy
# conda install seaborn
# conda install -c conda-forge statsmodels

# Y luego vamos a cargar los paquetes y datos que vamos a utilizar

import os # Para interactuar con el sistema de archivos el sistema operativo
import matplotlib.pyplot as plt # Para trabajar con gráficos y figuras

import pandas as pd 
import numpy as np 
import statistics
import scipy as sp # Evolución de NumPy en el que se le han añadido funciones para trabajar con distribuciones y estadística clásica
import seaborn as sns # Para trabajar con gráficos (más orientado a análisis de datos), como el paquete math plot lib, y el paquete de datos de test que utilizaremos
import random
import math

from statsmodels.stats.proportion import proportions_ztest # Desarrollar modelos basados en estadística clásica

# Por sencillez, y para hacer estos ejercicios le quitaremos todos los nulos

df = pd.read_csv('HistoricoVentas.csv', index_col = 0)
df.dropna(inplace = True)

print(df)
print(df.info()) # Sacamos alguna información del data set
print(df.head()) # No devuelve los primeros registros del dataset

print("***********")
print("EJERCICIO 1")
print("***********")

# Haz un conteo de frecuencias de cuantas ventas hay de cada categoría de producto.

print(df.Product_Category.value_counts())

# Ruta de la carpeta de Descargas en macOS
# carpeta_destino = f"/Users/{os.getlogin()}/Downloads"
carpeta_destino = os.path.expanduser("~/Downloads")

# Creamos la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
	os.makedirs(carpeta_destino)

df.Product_Category.value_counts().plot(kind = 'bar');

# Guardar el gráfico en la carpeta de destino
nombre_archivo = f"grafico_taller_ejercicios_estadística_ejercicio1.png"
ruta_grafico = os.path.join(carpeta_destino, nombre_archivo)
plt.savefig(ruta_grafico)
plt.close()  # Cerrar la figura para liberar memoria
print("Gráfico guardado en la carpeta de Descargas:", carpeta_destino)

print("***********")
print("EJERCICIO 2")
print("***********")

# ¿Cual es la categoría de producto más vendida? (saca la moda).

print(statistics.mode(df.Product_Category))

print("***********")
print("EJERCICIO 3")
print("***********")

# ¿Cuantos ventas ha habido de cada categoría de producto en cada almacén (Warehouse)? (haz una tabla cruzada)
# Consejo: como hay muchas más categorías que almacenes pon las categorías como filas. 

print(pd.crosstab(df.Product_Category,df.Warehouse,margins = True))

print("***********")
print("EJERCICIO 4")
print("***********")

# Ahora haz lo mismo pero que salga en porcentaje de las categorías de producto(que las filas sumen 100%)

# print(pd.crosstab(df.Product_Category,df.Warehouse,margins = True, normalize = "all")) # Es más interesante ver estos valores en %
print(pd.crosstab(df.Product_Category,df.Warehouse,margins = True, normalize = "index")) # Es más interesante ver estos valores en % (solo filas, normalize = "index")

print("***********")
print("EJERCICIO 5")
print("***********")

# ¿Está relacionada la venta de ciertas categorías de productos con ciertos almacenes? ¿o por el contrario se venden igual en todos?
# Comprúebalo calculando el pvalor de una prueba de chi-cuadrado (si es menor que 0.05 es que sí hay relación entre las dos variables).

# Al ser variables categóricas y no numéricas utilizamos el método del chi cuadrado

tabla = pd.crosstab(df.Product_Category,df.Warehouse)
chi, pvalor, gl, experado = sp.stats.chi2_contingency(tabla)

print(chi)
print(pvalor)
print("pvalor es {} por lo que al ser más pequeño que 0.05 si podemos afirmar que la venta de ciertas categorías de productos está relacionada con ciertos almacenes (es decir, rechazar la hipótesis nula que se venden igual en todos).".format(pvalor))

print("***********")
print("EJERCICIO 6")
print("***********")

# Usando la variable ventas (Order_Demand) calcula la media winsorizada un 5% por cada cola

# La media normal se llama media aritmétrica, y será la que usaremos en la mayoría de los casos.
# La media winsorizada se usa cuando hay valores atípicos que pueden sesgar el valor de la media.

# Para hacer la media de porcentajes o tasas hay que usar la media geométrica.
# Para hacer la media de medias hay que usar la media armónica.

print(statistics.mean(df.Order_Demand)) # La media aritmética

# Para calcular la media winsorizada
# Para lo que estamos calculando, usando la variable ventas (Order_Demand). 
# Si hay un valor que dispara la media pero no es representaivo cogeríamos el valor anterior y lo substituiríamos por ese
# En este caso, hay una categoría de producto (Category_019) que dispara la media del resto

order_demand = np.array(df.Order_Demand)

print(order_demand)
print(order_demand.mean())

# Winsorizar sustituye los valores fuera de los límites por el último valor
media = sp.stats.mstats.winsorize(order_demand, limits = [0.05, 0.05]) # Límite del 5% por encima, tal y como indica el enunciado
order_demand_w = media.mean()

print(order_demand_w)
print("La media winsorizada es: %.2f." %order_demand_w)

print("***********")
print("EJERCICIO 7")
print("***********")

# Usando la variable ventas (Order_Demand) calcula la mediana de las ventas

# Si se ordenan todos los valores de la variable en orden ascendente o descendente la medianta es el valor de la variable correpondiente al elemento que ocupa la posición central, es decir, el que está en el 50%.
# La mediana es una medida de centralización más recomendable que la media cuando tenemos distribuciones que no son normales, o cuando tenemos atípicos.
# En este ejemplo, cuando el orden de compras de estas  categorias de producto (en concreto, Category_019) tienen otro tipo de distribución (no normal)

order_demand_o = np.sort(order_demand) # Es importante que esten ordenados

print(order_demand_o)
mediana = order_demand_o
order_demand_median = statistics.median(media)
print(order_demand_median)
print("La mediana es: %.2f." %order_demand_median)

print("***********")
print("EJERCICIO 8")
print("***********")

# Ahora calcula la desviación típica y la varianza.

# La varianza es el resultado de restar la media a cada valor de la variable, elevarlo al cuadrado (para evitar los negativos), sumarlo todo, y dividir el resultado por el número de datos.
# Es una medida de dispersión, porque será mayor cuanto más lejos estén el global de los valores con respecto a la media.

order_demand = np.array(df.Order_Demand)
print(order_demand)
order_demand_median = statistics.median(mediana)
print(order_demand_median)

suma_cuadrados = sum((order_demand - order_demand_median) ** 2) # Sumatorio de (xi - xm)^2 / n
print(suma_cuadrados)
order_demand_var = suma_cuadrados / (len(order_demand))
print(order_demand_var)

# O si lo calculamos directamente:

order_demand_var = order_demand.var()
print("La varianza es: %.2f." %order_demand_var)

# El cálcul de la varianza implica elevar al cuadrado para quitarnos los negativos.
# Entonces, el dato obtenido ya no está en la misma escala que la media para poder compararlos, p.e.
# Como solución se hace la raiz cuadrada a la varianza para volver a traerla a la escala, y es lo que se llama desviación típica.

order_demand_std = order_demand.std()
print("La desviación típica es: %.2f." %order_demand_std)

print("***********")
print("EJERCICIO 9")
print("***********")

# Te proporciono dfr (de df recortado) con una muestra más manejable que incluya solo las ventas de la categoría de producto 019.

dfr = df[df.Product_Category == 'Category_019']
print(dfr)

print("************")
print("EJERCICIO 10")
print("************")

# Calcula el error típico de la variable Order_Demand a patir de dfr.

# En este data set de ventas no tiene mucho sentido calcularlo, ya que estamos trabajando con todos los datos.
# Estos cálculos tienen sentido cuando estamos trabajando con muestas de datos del total, aún asi:

# El error típico se cálcula obteniendo la desviación típica de esa muestra y dividiendolo por la raíz cuadada del tamaño de la muestra

muestra = np.array(dfr.Order_Demand) # La muestra, que en este caso es el total de la población
print(muestra)

desv_tip_muestra = np.array(muestra).std() # Calculamos la desviación típica
error_tipico = desv_tip_muestra / math.sqrt(22418) # O la len(dfr) que sería más correcto

print('El error típico según el teorema del límite central es: %f'%error_tipico)

print("************")
print("EJERCICIO 11")
print("************")

# Calcula el margen de error a un 95,5% de confianza.

# Se suelen usar los siguientes estándares:

# - 95.5% de nivel de confianza que equivale a 2 desviaciones típicas
# - 99.7% de nivel de confianza que equivale a 3 desviaciones típicas

# Pero recuerda, si incrementamos la confianza también incrementamos el error.

# Error muestral con nuestros datos para un nivel de confianza del 95,5%

error_95 = 2 * error_tipico
print(error_95)

print("************")
print("EJERCICIO 12")
print("************")

# Calcula el límite inferior y superior del intervalo de confianza a un 95,5% de confianza. (recuerda que deberás calcular también la media)

# Ahora que ya sabemos estimar el margen de error, calcular el intervalo de confianza es muy sencillo.

# El límite inferior vendrá dado por la media menos el margen de error.
# Y el límite superior vendrá dado por la media más el margen de error.

# Por tanto el intervalo entre el que estará el dato real, al nivel de confianza elegido vendrá dado por: [media - margen de error, media + margen de error]

#Intervalo de confianza en nuestro ejemplo a un NC del 95,5%
media_muestra = np.mean(muestra)

ic_95 = [media_muestra - error_95, media_muestra + error_95]
print(ic_95)

ic_95_superior = ic_95[1]
ic_95_inferior = ic_95[0]

# A la hora de intrepretarlo podríamos leerlo así:

# - Estamos un 95% seguros de que la media real va a estar entre 8514.85 y 9759.599
# Aunque técnicamente sería más correcto leerlo así:

# - Si repitiéramos 100 veces el experimento, en 95 de ellas nos saldría un dato entre 8514.85 y 9759.599

print('La media es: %.2f'%media_muestra)
print('El límite superior es: %.2f'%ic_95_superior)
print('El límite inferior es: %.2f'%ic_95_inferior)

print("************")
print("EJERCICIO 13")
print("************")

# Calcula qué porcentaje de los registros recogidos en dfr corresponden al producto 1359.

print(dfr.Product_Code.value_counts(normalize=True))

print("************")
print("EJERCICIO 14")
print("************")

# Calcula el intervalo de confianza de esa proporción a un 99,7% de confianza.
# (Consulta los apuntes para refrescar la fórmula del error típico en proporciones)

# INTERVALOS DE CONFIANZA EN PROPORCIONES

# Hemos visto como calcular los intervalos de confianza cuando estamos estimando medias.
# Pero el otro gran estadístico que podemos estar usando son las proporciones.

# El error típico para distribuciones muestrales de proporciones se calcular como:

# Raiz cuadrada de ((p * q) dividido por n), siendo:

# p = la proporción obtenida
# q = 1-p
# n = tamaño de la muestra

# Vamos a calcular su intervalo de confianza con un 99,7% de nivel de confianza.

p = 0.034570
q = 1 - p
n = len(dfr)

error_tipico_prop = math.sqrt((p * q)/ n)
error_tipico_prop_99 = error_tipico_prop * 3

ic_prop_99 = [p - error_tipico_prop_99, p + error_tipico_prop_99]

ic_prop_99_superior = ic_prop_99[1]
ic_prop_99_interior = ic_prop_99[0]

print('La proporción es: %.3f'%p)
print('El límite superior es: %.3f'%ic_prop_99_superior)
print('El límite inferior es: %.3f'%ic_prop_99_interior)

print("************")
print("EJERCICIO 15")
print("************")

# Recordemos que la media de ventas en dfr es de 9137$. Pero eso es la muestra que hemos podido recoger.
# Sin embargo el director comercial nos dice que realmente la media está en 9500$.
# ¿Nos podemos creer lo que dice con un nivel de seguridad del 95%?

# CONTRASTE DE MEDIAS EN LA POBLACIÓN

# Queremos ver si el valor de la media obtenido en la muestra puede ser compatible con un hipotético valor en la población.
# Para ello seguiremos la metodología explicada, usando como estadístico de contraste la prueba t.
# Usaremos la implementación de Scipy con la función ttest_1samp() ya que estamos usando el estadístico t en una sola muestra (no comparamos entre dos muestras)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html

# A esta función tenemos que pasarle:

# a: los datos
# popmean: el valor de h0 a contrastar

# Como ejemplo vamos a calcular si es posible que la media de ventas en la población sea de 9500$.

# Utilizo el CONTRASTE DE MEDIAS EN LA POBLACIÓN ya que el data set son todos los datos y queremos comprobarlo directamente con la población

# Primero recordamos cual era el valor en nuestra muestra
print(dfr.Order_Demand.mean())

# PASO 1: definimos las hipótesis. Por tanto h1 es que sea diferente de 9500
h0 = 9500

# PASO 2: elegimos un nivel de confianza que nos da un alpha, elegimos NC = 95%
alpha = 0.05

# PASO 3: calculamos el pvalor según el estadístico de contraste elegido (t sobre 1 muestra)
p_valor = sp.stats.ttest_1samp(a = dfr.Order_Demand, popmean = h0)[1]
print(f"{p_valor:.3f}") # Esto es para que no salga con notación científica

# Por tanto vemos que el pvalor es menor que alpha, por tanto no podemos aceptar la H0 y por tanto no es probable que la media de propinas en la población sea de 25$.

# pvalor < alpha NO. Si fuera menor si podemos rechazar la hipótesis nula, es decir, no considerar 9500$ como una media válida
print("pvalor es %.3f por lo que NO podemos rechazar la hipóstesis nula, es decir, que la media sea 9500$." %pvalor) # Me tengo que creer lo que dice el director comercial

print("************")
print("EJERCICIO 16")
print("************")

# Saca de nuevo la tabla cruzada de la categoría de productos cruzada con los almacenes (Warehouse), pero en este caso saca los porcentajes verticales (que las columnas sumen el 100%)

# Fíjate en la penetración de ventas de la categoría 006 en los almacenes A y S, es 0.028 y 0.036 respectivamente.

# ¿Podríamos pensar que esa diferencia es real y esa categoría se vende más en el S? ¿O es simplemente fruto de nuestros datos y no representa una diferencia real?
# Queremos estar muy seguros, así que contrástalo al 99% de confianza.

# CONTRASTE DE PROPORCIONES ENTRE DOS MUESTRAS

# Queremos ver si una diferencia en la proporción obtenida entre dos muestras puede ser estadísticamente significativa.
# Para ello seguiremos la metodología explicada, usando como estadístico de contraste la prueba z.

# Scipy no tiene implementación (hasta donde yo sé) para hacer contrastes de proporciones, así que vamos a usar la implementación del paquete statsmodels, con la función proportions_ztest() ya que estamos usando el estadístico z, y como son dos muestras, a los parámetros count y nobs les tendremos que pasar un array.

# https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html

# A esta función tenemos que pasarle:

# count: los éxitos (lo que queremos medir)
# nobs: el tamaño de la muestra
# value: el valor de la hipótesis nula a testar

# Como estamos haciendo el contraste de dos muestras le pasaremos un array con dos números a count y a nobs, y no usaremos el value.

# Como ejemplo vamos a calcular si la diferencia entre el porcentaje de fumadores entre hombres y mujeres puede ser significativa.

# Primero recordamos cual era el valor en nuestra muestra (en porcentaje)
print(pd.crosstab(df.Product_Category,df.Warehouse,margins = True, normalize = "columns")) # Es más interesante ver estos valores en % (solo filas, normalize = "index")

# Primero recordamos cual era el valor en nuestra muestra (en absoluto)
print(pd.crosstab(df.Product_Category,df.Warehouse,margins = True)) # Es más interesante ver estos valores en % (solo filas, normalize = "index")

# PASO 1: definimos las hipótesis
# H0: el porcentaje de ventas de la categoría 006 en los dos almacenes es igual en la población
# H1: el porcentaje de ventas de la categoría 006 en los dos almacenes no es igual en la población

# PASO 2: elegimos un nivel de confianza que nos da un alpha, elegimos NC = 99%
alpha = 0.01

# PASO 3: calculamos el pvalor según el estadístico de contraste elegido (z sobre dos muestras)

# Tenemos que pasar los datos en absoluto

exitos_a = 190
exitos_s = 152
muestra_a = 190 + 6523
muestra_s = 152 + 3971

# Lo pasamos a array porque así lo pide la función

array_exitos = np.array([exitos_a,exitos_s])
array_muestras = np.array([muestra_a,muestra_s])

p_valor = proportions_ztest(count = array_exitos, nobs = array_muestras)[1]
print(f"{p_valor:.3f}") #Esto es para que no salga con notación científica

# pvalor < alpha NO. Como es mayor si podemos rechazar la hipótesis nula, es decir, considerar que esa categoría de proudcto se vende más en el almacén A que el B
print("pvalor es %.3f (pvalor < 0.01) por lo que NO podemos rechazar la hipóstesis nula, es decir, considerar que esa categoría de proudcto se vende más en el almacén A que el B." %p_valor) 
