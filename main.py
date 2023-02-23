import random
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.callbacks import EarlyStopping


# Descarga los datos necesarios para el procesamiento de lenguaje natural de NLTK
nltk.download("punkt")
nltk.download("wordnet")

# Inicializa el lematizador de palabras
lemmatizer = WordNetLemmatizer()

# Carga el archivo JSON con los intents
with open("intents.json") as archivo:
    intents = json.load(archivo)

# Crea un diccionario de palabras y etiquetas
palabras = []
etiquetas = []
docs_x = []
docs_y = []

# Recorre cada intent
for intent in intents["intents"]:
    print(intent)
    # Recorre cada patrón del intent
    for patron in intent["patrones"]:
        print(patron)
        # Tokeniza el patrón
        wrds = nltk.word_tokenize(patron)
        # Agrega las palabras a la lista de palabras
        palabras.extend(wrds)
        # Agrega el par (patrón, etiqueta) a la lista de documentos
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    # Agrega la etiqueta a la lista de etiquetas si no está presente
    if intent["tag"] not in etiquetas:
        etiquetas.append(intent["tag"])

# Lematiza las palabras y las ordena alfabéticamente
palabras = sorted(list(set([lemmatizer.lemmatize(p.lower()) for p in palabras])))

# Ordena alfabéticamente las etiquetas
etiquetas = sorted(etiquetas)

# Inicializa las listas de entrenamiento y salida
entrenamiento = []
salida = []

# Crea una lista de salida vacía con la misma longitud que la lista de etiquetas
salida_vacia = [0] * len(etiquetas)

# if os.path.isfile("modelo_chatbot.h5"):
#     modelo = keras.models.load_model("modelo_chatbot.h5")
# else:
# Recorre cada documento y crea la bolsa de palabras correspondiente y la salida correspondiente a la etiqueta
for x, doc in enumerate(docs_x):
    # Inicializa la bolsa de palabras
    bolsa = []
    # Lematiza cada palabra del documento
    wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]
    # Crea una bolsa de palabras para el documento
    for w in palabras:
        bolsa.append(1) if w in wrds else bolsa.append(0)
    # Crea la salida correspondiente a la etiqueta
    fila_salida = list(salida_vacia)
    fila_salida[etiquetas.index(docs_y[x])] = 1
    # Agrega la fila de entrenamiento y la salida a sus respectivas listas
    entrenamiento.append(bolsa)
    salida.append(fila_salida)
# Convierte las listas de entrenamiento y salida en arrays numpy
entrenamiento = np.array(entrenamiento)
salida = np.array(salida)
# Crea el modelo de red neuronal
modelo = keras.Sequential(
    [
        layers.Dense(128, input_dim=entrenamiento.shape[1], activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(etiquetas), activation="softmax"),
    ]
)
# Compila el modelo
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Agrega una función de callback para detener el entrenamiento temprano si la precisión deja de mejorar
callback = EarlyStopping(monitor='accuracy', patience=10)
# Entrena el modelo
historial_entrenamiento = modelo.fit(entrenamiento, salida, epochs=1000, batch_size=16, verbose=1, callbacks=[callback])
# Muestra la precisión y la pérdida en cada época
for i, acc in enumerate(historial_entrenamiento.history['accuracy']):
    print("Epoch:", i+1, "- Precisión:", acc, "- Pérdida:", historial_entrenamiento.history['loss'][i])
# Guarda el modelo
modelo.save("modelo_chatbot.h5")


modelo = keras.models.load_model("modelo_chatbot.h5")

# Define el umbral de probabilidad
UMBRAL_PROBABILIDAD = 0.7

# Crea la función para predecir las etiquetas
def predecir_etiqueta(texto):
    # Crea una bolsa de palabras a partir del texto
    bolsa = [0] * len(palabras)
    wrds = nltk.word_tokenize(texto)
    wrds = [lemmatizer.lemmatize(w.lower()) for w in wrds]
    for w in wrds:
        for i, palabra in enumerate(palabras):
            if palabra == w:
                bolsa[i] = 1
    # Predecir las etiquetas con el modelo
    resultados = modelo.predict(np.array([bolsa]))[0]
    # Obtenemos las etiquetas con una probabilidad mayor al umbral
    etiquetas_posibles = [etiquetas[i] for i, p in enumerate(resultados) if p > UMBRAL_PROBABILIDAD]
    # Si no hay etiquetas con una probabilidad suficientemente alta, devolvemos una lista vacía
    if not etiquetas_posibles:
        return []
    return etiquetas_posibles

# Prueba la función predecir_etiqueta
while True:
    texto = input("Pregúntame algo: ")
    if texto == "salir":
        break
    etiquetas_posibles = predecir_etiqueta(texto)
    if etiquetas_posibles:
        print("Posibles preguntas:")
        for etiqueta in etiquetas_posibles:
            print("- " + etiqueta)
    else:
        print("Lo siento, no entendí la pregunta.")