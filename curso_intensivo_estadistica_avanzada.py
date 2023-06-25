print("********************")
print("ESTADÍSTICA AVANZADA")
print("********************")

# - Conceptos más avanzados que estaremos usando sobre todo cuando hagamos modelización estadística
# - Errores de interpretación en el día a día de las conclusiones que se sacan en una empresa, pero que no son correctas.

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

df = sns.load_dataset("tips")

print(df.info()) # Sacamos alguna información del data set
print(df.head()) # No devuelve los primeros registros del dataset

print("****************************")
print("VALOR TEÓRICO Y ERROR MEDIDA")
print("****************************")

# El valor teórico es una combinación lineal de variables con ponderaciones estimadas empíricamente.
# Es decir, cuando hagamos modelos predictivos que vienen del campo de la estadística, como regresiones, lo que estará prediciendo el modelo es el valor teórico.
# Pero el valor teórico se puede descomponer en dos componentes: la parte "verdadera" de la variable objetivo que se puede predecir directamente a través de las predictoras, y un error.
# Ese error se llama error de medida y es lo que siempre estaremos intentando minimizar.

# A nivel operativo ese error tendrá dos fuentes principales.

# - Una es la falta de datos.
# - Y la otra es la falta de adecuación del algoritmo que estemos usando.

# Por tanto en Data Science siempre mejoraremos:

# - Incorporando nuevos y/o mejores datos.
# - Utilizando algoritmos más apropiados al problema.

# Y a nivel conceptual ese error puede tener dos fuentes, que son imporantes para detectar malas prácticas en la empresa:

# - La validez: que estemos midiendo lo que en realidad queremos medir.
# - La fiabilidad: que lo estemos midiendo con los instrumentos adecuados.

# Por ejemplo un error de validez en la empresa es cuando las agencias intentan convencer a los clientes de que invertir en campañas de "LIKES" mejorará sus resultados comerciales.
# Y un ejemplo de error de fiabilidad es intentar medir la importancia de cada canal en nuestro marketing mediante informes basados en atribución "LAST CLICK".

print("**********************")
print("SUPUESTOS ESTADÍSTICOS")
print("**********************")

# Las técnicas que utilizaremos en la parte de modelización vienen de dos campos:

# - Machine Learning: como árboles de decisión, random forest, redes neuronales, entre otros.
# - De la estadística: como regresión múltiple o regresión logística.

# Estas últimas fueron creadas bajo una serie de supuestos, que idealmente deberían cumplirse para que se pueda utilizar la técnica.
# Es más, los supuestos deberían cumplirse tanto a nivel individual de cada variable como en el valor teórico combinado.
# Pero en la práctica es muy raro que estos supuestos se cumplan, y realmente las técnicas han demostrado ser robustas a las violaciones de los mismos.

# Hay que aprender la forma de evaluar los modelos en la práctica para ver si, aún sin cumplir totalmente los supuestos, son modelos útiles o no.
# Entenderlos a nivel conceptual nos llevará a entender mucho mejor este tipo de técnicas y a ser capaz de mejorar su capacidad predictiva.

# Los supuestos más importantes son:

# - Normalidad
# - Heterocedasticidad de varianzas
# - Linealidad
# - Multicolinealidad

# NORMALIDAD

# Conociendo el concepto de distribución normal:

# Si la variación con respecto a una normal es suficientemente grande, entonces todos los test estadísticos resultantes (que se basarán en estadísticos como t o F) no serán válidos.
# Medir la normalidad multivariante es complicado, por lo que la aproximación práctica suele ser medir y corregir la normalidad univariante de todas las variables que formarán el valor teórico.

# Formas de identificarla: La normalidad se puede evaluar a nivel gráfico con gráficos como el histograma o el Q-Q.
# Solución: Transoformaciones de la variable para hacerla más normal: inversa, cuadrado, raiz cuadrada, logaritmo, entre otras.

#Ejemplo de un qqplot

import matplotlib.pyplot
import scipy.stats

ejemplo = np.random.normal(loc = 20, scale = 5, size=50)

scipy.stats.probplot(ejemplo, dist="norm", plot=matplotlib.pyplot)

# Ruta de la carpeta de Descargas en macOS
# carpeta_destino = f"/Users/{os.getlogin()}/Downloads"
carpeta_destino = os.path.expanduser("~/Downloads")

# Creamos la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
	os.makedirs(carpeta_destino)

# Guardar el gráfico en la carpeta de destino
nombre_archivo = f"grafico_supuestos_normalidad.png"
ruta_grafico = os.path.join(carpeta_destino, nombre_archivo)
plt.savefig(ruta_grafico)
plt.close()  # Cerrar la figura para liberar memoria
print("Gráfico guardado en la carpeta de Descargas:", carpeta_destino)

# HETEREOCEDASTICIDAD DE VARIANZAS

# Significa que la varianza de la variable objetivo no es constante en el recorrido de la variable predictora.
# Es decir, que para unos valores la predicción será más precisa que para otros.

# Formas de identificarla: Para variables contínuas: con diagramas de dispersión
# Analizando el gráfico de los residuos (diferencias entre valor predicho y valor real)

# Solución: En la mayoría de casos es causada por la no normalidad, por lo que corrigiendo la normalidad se corrige también la heterocedasticidad.

# LINEALIDAD

# Se refiere a que exista una relación lineal de cada variable predictora con la target.
# Es un supuesto para todas las técnicas que se basen de una u otra forma en la correlación, como la regresión múltiple o la regresión logística.

# Formas de identificarla:

# - Hacer la matriz de correlaciones de cada predictora con la target
# - Gráficos de dispersión de cada predictora con la target
# - Análisis de residuos del modelo. Cualquier pauta no lineal visible será la que las variables no han podido explicar linealmente

# Solución:

# - Linealizar las relaciones mediante transformación de las variables originales
# - Usar algoritmos no lineales

# MULTICOLINEALIDAD

# Se refiere a que exista correlación entre las variables predictoras.
# Los modelos que usamos normalmente (aditivos) asumen independencia entre las variables predictoras.

# Si eso no se cumple pasará que:

# - Podemos estar sobreponderando conceptos
# - Podemos causar efectos extraños y variaciones en los modelos: exponentes desproporcionados, signos invertidos, o incluso no convergencia

# Formas de identificarla:

# - Hacer la matriz de correlaciones entre las variables predictoras
# - Identificar durante el desarrollo del modelo variables que apriori deberían predecir pero salen como no predictoras (por la correlación parcial aportada por otras variables)

# Solución:

# - No meter variables correlacionadas
# - Aplicar reducción de variables como Componentes Principales (poco uso en la realidad)

print("*************")
print("BOOTSTRAPPING")
print("*************")

# O su traducción como re-muestreo.
# El remuestreo coneguía demostrar el teorema del límite central y cómo se puede utilizar utilizando la estadística inferencial.
# Pero tiene más propiedades. Y una de ellas es la capacidad de generalización ya que uno de los inconvenientes que tendremos en los modelos es el sobre ajuste.
# El caso es que hay algoritmos que se basan precisamente en bootstrapping para conseguir evitar ese sobreajuste y ser capaz de predecir mucho mejor ante datos que no han visto nunca.
# Lo usan por ejemplo el bagging, que viene de bootstrap aggregating, y básicamente consiste en remuestrear casos y / o variables, hacer un modelo sobre cada una de esas muestras y luego agregarlos para hacer la predicción final.
# O por ejemplo el Random Forest, que se basa en este principio y es de los algoritmos más estables.

print("***************************")
print("PENETRACIÓN VS DISTRIBUCIÓN")
print("***************************")

# No es estrictamente un concepto estadístico, pero podemos meterlo dentro de los análisis que más se confunden.

# Este efecto está implicado en falacias como: "Según la DGT el alcohol está implicado en el 30% de los accidentes mortales".
# Luego el alcohol no está implicado en el 70% de los accidentes mortales.
# Por tanto podemos concluir que es más peligroso conducir sobrio que borracho.

# O su equivalente muy frecuente en el mundo de la empresa: "El 70% de nuestras compras las hacen clientes varones entre 30 y 45 años, por tanto son los más compradores"
# No será cierto si ese perfil fuera el 80% del total de clientes por ejemplo.

# En general hay que diferenciar muy bien estos dos conceptos:

# - Penetración: qué porcentaje de una base determinada presenta la característica a analizar
# - Distribución: qué porcentaje representa un sobconjunto sobre el total

# El mejor truco para diferenciarlo es que al sumar penetraciones no tiene por qué sumar 100%.
# Pero al sumar distribuciones sí tiene que sumar 100%.

# Por ejemplo, decir que el 70% de nuestros clientes son mujeres es un análisis de distribución, y tiene que sumar 100% con el 30% restante que son hombres.
# Sin embargo decir que el producto A lo compra el 70% de las mujeres y el 50% de los hombres es un análisis de penetración, y no tiene por qué sumar 100%.

# Estos dos son conceptos absolutamente claves cuando estemos haciendo análisis de perfilado, insights o segmentaciones.

print("********************")
print("ABSOLUTO VS RELATIVO")
print("********************")

# 3 fuentes de error:

# - Dar el absoluto sin dar el total.
# - Dar el porcentaje sin dar el término de comparación absoluto.
# - no tener en cuenta las escalas de los gráficos.

# Con esto muchas veces se intenta hacer trampas, usando uno u otro según convenga al mensaje que se quiere mandar.

# Por ejemplo dar un dato en absoluto, sin dar el total, hará que su efecto parezca mayor o menor.
# Decir que este fin de semana han muerto 50 personas en carretera parece mucho. Pero si decimos que ha sido salida de Semana Santa, con 15.000.000 millones de desplazamientos el dato queda matizado.

# O al revés, dar el dato en relativo a sabiendas de que la base original es muy pequeña y por tanto parecerá un efecto mayor.
# Por ejemplo si decimos que esta semana hemos vendido un 300% más que la anterior parece mucho, pero si la anterior sólo vendimos un producto el dato ya no impresiona tanto.

# O jugar con los ejes para que en las comparaciones se perciba un efecto diferente al real, como hacen muchos medios de comunicación en redes sociales.

# Solución:

# - No considerar datos en porcentaje que tengan una base menor a 50 casos (p.e. las unidades de venta)
# - Siempre poner en contexto ambas dimensiones (si tienes dato absoluto, ponerlo en relativo y al revés, p.e. accidentes de tráfico)
# - Cuidado con los ejes y las posiciones relativas o absolutas (p.e. el gráfico de los medios de comunicación)
