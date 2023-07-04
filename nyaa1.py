import numpy as np
import matplotlib.pyplot as plt
import random

direcciones = np.array([
    [-1, 1],  # Noroeste, Azul
    [0, 1],   # Norte, Verde
    [1, 1],   # Noreste, Rojo
    [-1, 0],  # Oeste, Cian
    [0, 0],   # Sin movimiento, Magenta
    [1, 0],   # Este, Amarillo
    [-1, -1], # Suroeste, Negro
    [0, -1],  # Sur, Púrpura
    [1, -1]   # Sureste, Naranja
])

colores =  ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange']

def obtener_color(cromosoma):
    max_index = np.argmax(cromosoma)
    return colores[max_index]

def generar_cromosoma():
    return np.random.uniform(0, 1, 9)

def normalizar_cromosoma(cromosoma):
    suma_total = np.sum(cromosoma)
    normalizado = cromosoma / suma_total
    return normalizado

def generar_puntos_iniciales():
    return int(np.random.uniform(1, tam * 0.8))

class Individuo:
    def __init__(self, cromosoma, x, y):
        self.cromosoma = normalizar_cromosoma(cromosoma)
        self.color = obtener_color(self.cromosoma)
        self.x = x
        self.y = y
        self.fitness = 0
        self.steps = 0

    def evaluar_fitness(self, steps):
        return 1 / steps if steps != 0 else 1 

    def mutar(self):
        if random.random() < tasa_mutacion:
            index = random.randint(0, len(self.cromosoma) - 1)
            self.cromosoma[index] = random.random()
            self.cromosoma = normalizar_cromosoma(self.cromosoma)

class Poblacion:
    def __init__(self, tamaño):
        self.individuos = []
        self.finalizadores = []
        posiciones_ocupadas = set()

        for _ in range(tamaño):
            x, y = generar_puntos_iniciales(), generar_puntos_iniciales()

            while (x, y) in posiciones_ocupadas:
                x, y = generar_puntos_iniciales(), generar_puntos_iniciales()

            posiciones_ocupadas.add((x, y))
            self.individuos.append(Individuo(cromosoma=generar_cromosoma(), x=x, y=y))

    def mover(self):
        individuos_restantes = []

        for individuo in self.individuos:
            dirección = np.random.choice(range(len(direcciones)), p=individuo.cromosoma)
            nueva_x = individuo.x + direcciones[dirección][0]
            nueva_y = individuo.y + direcciones[dirección][1]

            if individuo.x != tam:
                if 1 <= nueva_x <= tam and 1 <= nueva_y <= tam:
                    if not any(
                        otro_individuo.x == nueva_x and otro_individuo.y == nueva_y
                        for otro_individuo in self.individuos + self.finalizadores
                    ):
                        individuo.x = nueva_x
                        individuo.y = nueva_y

            if individuo.x == tam:
                self.finalizadores.append(individuo)
                individuo.steps = _ + 1
                individuo.fitness = individuo.evaluar_fitness(individuo.steps)
            else:
                individuos_restantes.append(individuo)

        self.individuos = individuos_restantes

    def seleccion_torneo(self):
        torneo = self.finalizadores
        torneo.sort(key=lambda x: x.fitness, reverse=True)
        p = 0.3
        probabilidades_acumuladas = [p * (1 - p) ** (i - p) for i in range(len(torneo))]
        probabilidades_acumuladas = normalizar_cromosoma(probabilidades_acumuladas)
        mejor = np.random.choice(torneo, p=probabilidades_acumuladas)
        return mejor

    def cruce(self, padre1, padre2):
        cruce_binario = generar_cromosoma()
        cromosoma_hijo1, cromosoma_hijo2 = [], []

        for i in range(len(cruce_binario)):
            cruce_binario[i] = int(cruce_binario[i])
            if i < 1:
                cromosoma_hijo1.append(padre1.cromosoma[i])
                cromosoma_hijo2.append(padre2.cromosoma[i])
            else:
                cromosoma_hijo1.append(padre2.cromosoma[i])
                cromosoma_hijo2.append(padre1.cromosoma[i])

        cromosoma_hijo1 = normalizar_cromosoma(cromosoma_hijo1)
        cromosoma_hijo2 = normalizar_cromosoma(cromosoma_hijo2)

        hijo1 = Individuo(cromosoma_hijo1, generar_puntos_iniciales(), generar_puntos_iniciales())
        hijo2 = Individuo(cromosoma_hijo2, generar_puntos_iniciales(), generar_puntos_iniciales())

        return hijo1, hijo2

    def crear_nueva_generacion(self):
        nueva_generación = self.finalizadores

        while len(nueva_generación) < tamaño_población:
            padre1 = self.seleccion_torneo()
            padre2 = self.seleccion_torneo()

            while padre1 == padre2:
                padre2 = self.seleccion_torneo()

            hijo1, hijo2 = self.cruce(padre1, padre2)
            nueva_generación.append(hijo1)
            nueva_generación.append(hijo2)

        posiciones_ocupadas = set()

        for individuo in nueva_generación:
            individuo.mutar()
            x, y = generar_puntos_iniciales(), generar_puntos_iniciales()

            while (x, y) in posiciones_ocupadas:
                x, y = generar_puntos_iniciales(), generar_puntos_iniciales()

            individuo.x = x
            individuo.y = y
            posiciones_ocupadas.add((x, y))

        self.individuos = nueva_generación
        self.finalizadores = []

tamaño_población = 50
generaciones = 10
tasa_mutacion = 0.05
tam = 30
max_steps = int(tamaño_población * 1.5)
finalizadores_por_generación = [0] * generaciones

población = Poblacion(tamaño_población)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel("Eje x")
plt.ylabel("Eje y")
plt.xlim(0, tam)
plt.ylim(0, tam)


plt.ion()

for i in range(1, generaciones + 1):
    ax.set_title(f"Generación: {i}")

    for _ in range(max_steps):
        población.mover()

    if i > 1:
        sc.remove()  # Borra los puntos anteriores

    todos_los_individuos = población.individuos + población.finalizadores

    sc = ax.scatter(
        [individuo.x for individuo in todos_los_individuos],
        [individuo.y for individuo in todos_los_individuos],
        c=[individuo.color for individuo in todos_los_individuos],
        marker='s',
        s=50
    )

    plt.draw()
    plt.pause(0.7)

    finalizadores_por_generación[i - 1] = len(población.finalizadores)

    población.crear_nueva_generacion()

plt.ioff()

plt.figure()
plt.plot(range(1, generaciones + 1), finalizadores_por_generación, marker="o")
plt.xlabel("Generación")
plt.ylabel("Número de Finalizadores")
plt.title("Número de Finalizadores por Generación")
plt.show()


plt.waitforbuttonpress()
plt.show(block=True)
