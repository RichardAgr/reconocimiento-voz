import pygame
import sys
import threading
import speech_recognition as sr
import heapq
import tensorflow as tf
import numpy as np
# Tamaño de la ventana y de los bloques del mapa
WIDTH = 914
HEIGHT = 700
BLOCK_SIZE = 20
window_frase=""
#command2='INICIO'
# Colores
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)

# Mapa de la ciudad
city_map = [
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Nombres de las calles con sus coordenadas de inicio y fin
#suponiendo que (y,x) ordenadas asi las coordenadas
street_names = {
    ((0, 4), (19, 4)): "agente juan camine a la calle profe inolvidable", #
    ((19, 4), (0, 4)): "agente juan camine a la calle final profe inolvidable", #
    ((5, 0), (5, 16)): "agente juan camine a la calle del adjetivo",#
    ((5, 16), (5, 28)): "agente juan camine a la  calle del si le",#
    ((0, 23), (19, 23)): "agente juan camine a la calle del sustantivo",#
    ((0, 28), (19, 28)): "agente juan camine a la calle de los errores",#
    ((10, 9), (19,9)): "agente juan camine a la calle de ser y estar",#
    ((0,16), (17, 16)): "agente juan camine a la avenida hablo español",#
    ((10,28), (10, 10)): "agente juan camine a la calle de los deberes hechos",#
    ((19, 18), (19, 29)): "agente juan camine a la avenida del indicativo",#
    ((19, 0), (19, 14)): "agente juan camine a la avenida del subjuntivo ",#
    ((21, 4), (44, 4)): "agente juan camine a la calle del vocabulario", #
    ((19, 9), (44, 9)): "agente juan camine a la calle del instituto cervantes",#
    ((21, 16), (41, 16)): "agente juan camine a la avenida profedeele",#
    ((30, 16), (30, 28)): "agente juan camine a la calle de los verbos",#
    ((35, 16), (35, 28)): "agente juan camine a la calle de la gramática",
    ((38, 0), (38, 9)): "agente juan camine a la calle de las dudas",#
    ((30, 23), (44, 23)): "agente juan camine a la calle del me gusta",#
    ((41, 16), (43, 9)): "agente juan camine a la calle de la ñ",#
    ((30, 28), (44, 28)): "agente juan camine a la calle de por y para",
    ((41,16), (41, 16)): "agente juan camine al monumento nivel c2",
    ((8,4), (8, 4)): "agente juan camine a la lavandería",
    ((41,4), (41, 4)): "agente juan camine a la gasolinera",
    ((24,16), (24, 16)): "agente juan camine al banco",
    ((15,28), (15, 28)): "agente juan camine a la farmacia",
    ((41,23), (41, 23)): "agente juan camine a la estacion de tren",
    ((2,23), (2, 23)): "agente juan camine a la estacion de bomberos",
    ((32,23), (32, 23)): "agente juan camine a la universidad",
    ((14,9), (14, 9)): "agente juan camine a la tienda de instrumentos",
    ((25,4), (25, 4)): "agente juan camine a la panadería",
    ((41,9), (41, 9)): "agente juan camine a la pizzería",
    ((7,16), (7, 16)): "agente juan camine a la herboristería",
    ((7,28), (7, 28)): "agente juan camine al centro comercial",
    ((22,9), (22, 9)): "agente juan camine a la sala de conciertos",
    ((23,23), (23, 23)): "agente juan camine al parque ele",
    ((22,4), (22, 4)): "agente juan camine al restaurante",
    ((2,16), (2, 16)): "agente juan camine a la cafetería",
    ((19,11), (19, 11)): "agente juan camine a la iglesia",
    ((38,23), (38, 23)): "agente juan camine al bar",
    ((38,28), (38, 28)): "agente juan camine a la casa de pepe",
    ((32,28), (32, 28)): "agente juan camine a la tienda de ropa",
    ((12,23), (12, 23)): "agente juan camine al hospital",
    ((16,4), (16, 4)): "agente juan camine al supermercado",
    ((36,9), (36, 9)): "agente juan camine a la academia de idiomas",
    ((38,16), (38, 16)): "agente juan camine a la comisaria de policia"
}


# Inicializar Pygame
pygame.init()

# Crear la ventana
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interfaz de Mapa")

# Cargar la imagen del mapa
map_image = pygame.image.load("image/mapa.jpg")

# Cargar la imagen del agente
agent_image = pygame.image.load("image/person.png")
agent_rect = agent_image.get_rect()

agent_imageLeft = pygame.image.load("image/personLeft.png")
agent_rect = agent_image.get_rect()
# Crear la ventana
WINDOW_SIZE = (WIDTH, HEIGHT)
window = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Interfaz de Mapa")

# Coordenadas iniciales del agente
agent_x = 0
agent_y = 4
posX=agent_x
# Velocidad de desplazamiento del agente
move_speed = BLOCK_SIZE

clock = pygame.time.Clock()

# Configurar el reconocimiento de voz
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000  # Ajusta el umbral de energía según tus necesidades

# Función para capturar el comando de voz en un hilo separado
def capture_voice_command():
    global agent_x, agent_y, window_frase
    while True:
        with sr.Microphone() as source:
            print("Di el comando:")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio, language="es-ES").lower()
            print("Comando reconocido:", command)
            frase=obtener_frase_similar(command)
            move_to_street(frase)
            window_frase=frase
        except sr.UnknownValueError:
            print("No se pudo reconocer el comando de voz")
        except sr.RequestError as e:
            print("Error al solicitar el servicio de reconocimiento de voz; {0}".format(e))

# Función para calcular la ruta más corta utilizando el algoritmo A*
def calculate_shortest_path(start, end):
    open_list = []
    closed_list = set()

    # Crear un diccionario para almacenar los costos de los nodos
    g_scores = {start: 0}

    # Crear una cola de prioridad para almacenar los nodos a explorar
    heapq.heappush(open_list, (0, start))

    # Crear un diccionario para almacenar los padres de los nodos
    parents = {}

    while open_list:
        current_node = heapq.heappop(open_list)[1]

        if current_node == end:
            # Construir el camino desde el nodo final hasta el nodo inicial
            path = []
            while current_node in parents:
                path.insert(0, current_node)
                current_node = parents[current_node]
            return path

        closed_list.add(current_node)

        neighbors = get_neighbors(current_node)
        for neighbor in neighbors:
            if neighbor in closed_list:
                continue

            g_score = g_scores[current_node] + 1

            if neighbor not in [node[1] for node in open_list]:
                heapq.heappush(open_list, (g_score, neighbor))
            elif g_score >= g_scores[neighbor]:
                continue

            parents[neighbor] = current_node
            g_scores[neighbor] = g_score

    return None

# Función para obtener los vecinos válidos de un nodo en el mapa
def get_neighbors(node):
    x, y = node
    neighbors = []
    if x > 0 and city_map[y][x-1] == 0:
        neighbors.append((x-1, y))
    if x < len(city_map[0])-1 and city_map[y][x+1] == 0:
        neighbors.append((x+1, y))
    if y > 0 and city_map[y-1][x] == 0:
        neighbors.append((x, y-1))
    if y < len(city_map)-1 and city_map[y+1][x] == 0:
        neighbors.append((x, y+1))
    return neighbors

# Función para mover el agente a lo largo de la ruta hacia la calle destino
def move_to_street(street_name):
    global agent_x, agent_y, posX
    destination_coords = None
    for street_coords, name in street_names.items():
        if name.lower() == street_name:
            destination_coords = street_coords
            break

    if destination_coords is not None:
        start = (agent_x, agent_y)
        end = destination_coords[0]
        path = calculate_shortest_path(start, end)

        if path is not None:
            for node in path:
                posX=agent_x                 
                agent_x, agent_y = node                
                pygame.time.wait(100)  # Pausa de medio segundo entre movimientos
                pygame.event.pump()
                pygame.display.update()


def obtener_frase_similar(frase_entrada):
    # Frases almacenadas
    frases_almacenadas = [
    "agente juan camine a la calle profe inolvidable",
    "agente juan camine a la calle final profe inolvidable", 
    "agente juan camine a la calle del adjetivo",
    "agente juan camine a la  calle del si le",
    "agente juan camine a la calle del sustantivo",
    "agente juan camine a la calle de los errores",
    "agente juan camine a la calle de ser y estar",
    "agente juan camine a la avenida hablo español",
    "agente juan camine a la calle de los deberes hechos",
    "agente juan camine a la avenida del indicativo",
    "agente juan camine a la avenida del subjuntivo ",
    "agente juan camine a la calle del vocabulario", 
    "agente juan camine a la calle del instituto cervantes",
    "agente juan camine a la avenida profedeele",
    "agente juan camine a la calle de los verbos",
    "agente juan camine a la calle de la gramática",
    "agente juan camine a la calle de las dudas",
    "agente juan camine a la calle del me gusta",
    "agente juan camine a la calle de la ñ",
    "agente juan camine a la calle de por y para",
    "agente juan camine al monumento nivel c2",
    "agente juan camine a la lavandería",
    "agente juan camine a la gasolinera",
    "agente juan camine al banco",
    "agente juan camine a la farmacia",
    "agente juan camine a la estacion de tren",
    "agente juan camine a la estacion de bomberos",
    "agente juan camine a la universidad",
    "agente juan camine a la tienda de instrumentos",
    "agente juan camine a la panadería",
    "agente juan camine a la pizzería",
    "agente juan camine a la herboristería",
    "agente juan camine al centro comercial",
    "agente juan camine a la sala de conciertos",
    "agente juan camine al parque ele",
    "agente juan camine al restaurante",
    "agente juan camine a la cafetería",
    "agente juan camine a la iglesia",
    "agente juan camine al bar",
    "agente juan camine a la casa de pepe",
    "agente juan camine a la tienda de ropa",
    "agente juan camine al hospital",
    "agente juan camine al supermercado",
    "agente juan camine a la academia de idiomas",
    "agente juan camine a la comisaria de policia"
    ]

    # Preprocesamiento de texto
    vocabulario = set(" ".join(frases_almacenadas).lower().split())

    # Construir el índice del vocabulario
    vocabulario_indices = {palabra: i for i, palabra in enumerate(vocabulario)}
    indices_vocabulario = {i: palabra for i, palabra in enumerate(vocabulario)}

    # Convertir las frases almacenadas en vectores numéricos
    frases_almacenadas_encoded = []
    for frase in frases_almacenadas:
        frase_encoded = [0] * len(vocabulario)
        for palabra in frase.lower().split():
            if palabra in vocabulario_indices:
                frase_encoded[vocabulario_indices[palabra]] = 1
        frases_almacenadas_encoded.append(frase_encoded)

    frases_almacenadas_encoded = np.array(frases_almacenadas_encoded)

    # Crear y entrenar la red neuronal
    input_shape = (len(vocabulario),)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=input_shape, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(frases_almacenadas), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Convertir las frases almacenadas en etiquetas one-hot
    etiquetas_encoded = np.eye(len(frases_almacenadas))

    # Entrenar la red neuronal
    model.fit(frases_almacenadas_encoded, etiquetas_encoded, epochs=100, verbose=0)

    # Convertir la frase de entrada en un vector numérico
    frase_entrada_encoded = [0] * len(vocabulario)
    for palabra in frase_entrada.lower().split():
        if palabra in vocabulario_indices:
            frase_entrada_encoded[vocabulario_indices[palabra]] = 1

    # Predecir la frase más similar utilizando la red neuronal
    predicciones = model.predict(np.array([frase_entrada_encoded]))
    indice_frase_similar = np.argmax(predicciones)
    frase_similar = frases_almacenadas[indice_frase_similar]
    print("Frase similar: ",frase_similar)
    return frase_similar

# Crear y ejecutar el hilo para la captura de voz
voice_thread = threading.Thread(target=capture_voice_command)
voice_thread.daemon = True
voice_thread.start()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Dibujar el mapa en la ventana
    window.fill(WHITE)
    for y in range(len(city_map)):
        for x in range(len(city_map[0])):
            if city_map[y][x] == 1:
                pygame.draw.rect(window, GRAY, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            else:
                pygame.draw.rect(window, WHITE, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    font = pygame.font.Font(None, 24)
    text_window="Frase mejor aproximación: "+window_frase
    text3 = font.render(text_window, True, BLACK)
    text_rect3 = text3.get_rect(center=(WIDTH // 2, HEIGHT - 80))
    window.blit(text3, text_rect3)

    # Dibujar el agente en la ventana
    #pygame.draw.rect(window, BLACK, (agent_x * BLOCK_SIZE, agent_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    agent_x = max(0, min(agent_x, WIDTH - agent_rect.width))
    agent_y = max(0, min(agent_y, HEIGHT - agent_rect.height))
    window.blit(map_image, (0, 0))
    
    if posX<=agent_x:
        window.blit(agent_image, (agent_x* BLOCK_SIZE, agent_y* BLOCK_SIZE))        
    else:
        window.blit(agent_imageLeft, (agent_x* BLOCK_SIZE, agent_y* BLOCK_SIZE))
       
    # Mostrar el nombre de la calle si el agente se encuentra en una calle con nombre
    for street_coords, street_name in street_names.items():
        (x_start, y_start), (x_end, y_end) = street_coords
        if x_start <= agent_x <= x_end and y_start <= agent_y <= y_end:
            street_name = street_name.replace("agente juan camine a la", "")  # Eliminar la cadena no deseada
            font = pygame.font.Font(None, 24)
            text = font.render(street_name, True, BLACK)
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT - 20))
            window.blit(text, text_rect)
            break
    
    pygame.display.flip()
    # Controlar la velocidad de fotogramas
    clock.tick(60)  # Ajusta la velocidad de fotogramas según tus necesidades
