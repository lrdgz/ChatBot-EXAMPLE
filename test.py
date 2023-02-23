import json
import numpy as np
import tensorflow as tf

# Variables de configuración
CONFIG_FILE = "config.json"  # Archivo de configuración
INTENTS_FILE = "intents.json"  # Archivo de intenciones
THRESHOLD = None  # Umbral de probabilidad (se establecerá desde la configuración)

# Carga de la configuración
with open(CONFIG_FILE) as f:
    config = json.load(f)
    THRESHOLD = config.get("THRESHOLD", 0.5)

# Carga de las intenciones
with open(INTENTS_FILE) as f:
    intents = json.load(f)

# Función para crear la bolsa de palabras
def create_bag_of_words(sentence, words):
    # Tokenización de la oración
    sentence_words = sentence.split()
    # Inicialización de la bolsa de palabras
    bag = np.zeros(len(words), dtype=np.float32)
    # Conteo de las palabras de la oración en la bolsa de palabras
    for sw in sentence_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = 1
    return bag

# Función para predecir etiquetas
def predict_tag(model, words, sentence):
    # Creación de la bolsa de palabras
    bag = create_bag_of_words(sentence, words)
    # Predicción de la etiqueta
    results = model.predict(np.array([bag]))[0]
    # Selección de la etiqueta con mayor probabilidad
    prediction = {"tag": intents["tags"][np.argmax(results)], "probability": str(np.max(results))}
    # Verificación del umbral de probabilidad
    if float(prediction["probability"]) < THRESHOLD:
        prediction["tag"] = "desconocido"
    return prediction

# Creación del modelo
if config.get("model") == "neural_network":
    # Configuración del modelo
    model_config = config.get("model_config", {})
    model_architecture = model_config.get("architecture", [])
    model_loss = model_config.get("loss", "categorical_crossentropy")
    model_optimizer = model_config.get("optimizer", "adam")
    # Creación del modelo
    model = tf.keras.Sequential()
    for i, layer in enumerate(model_architecture):
        if i == 0:
            model.add(tf.keras.layers.Dense(layer["units"], input_shape=(len(intents["words"]),), activation=layer["activation"]))
        else:
            model.add(tf.keras.layers.Dense(layer["units"], activation=layer["activation"]))
    model.compile(loss=model_loss, optimizer=model_optimizer, metrics=["accuracy"])
    # Entrenamiento del modelo
    model.fit(np.array(intents["bags"]), np.array(intents["labels"]), epochs=config.get("epochs", 1000), batch_size=config.get("batch_size", 8))
    # Guardado del modelo
    model.save(config.get("model_file", "model.h5"))

# Carga del modelo
elif config.get("model") == "file":
    try:
      with open(config["model_file"], "rb") as f:
          model = tf.keras.models.load_model(f)
    except:
        print("¡Error al cargar el modelo! Se usará el modelo predeterminado.")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(len(intents["words"]),), activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(len(intents["tags"]), activation="softmax")
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.load_weights("default_weights.h5")

# Inicio del programa
print("¡Bienvenido al chatbot de soporte técnico!")
while True:
    # Lectura de la entrada del usuario
    sentence = input("Tu: ")
    # Predicción de la etiqueta
    prediction = predict_tag(model, intents["words"], sentence)
    # Selección de la respuesta
    for intent in intents["intents"]:
      if intent["tag"] == prediction["tag"]:
        print("Chatbot: " + np.random.choice(intent["responses"]))
        break