import random
import math
import matplotlib.pyplot as plt

V0 = 50
h = 100
g = 9.8

def calculate_distance(alpha):
    return (V0 * math.cos(alpha)) * (
    (V0 * math.sin(alpha) + math.sqrt((V0 * math.sin(alpha))**2 + 2 * g * h)) / g)

def draw_plot(alpha):
    # Oblicz czas lotu
    t_max = (V0 * math.sin(alpha) + math.sqrt((V0 * math.sin(alpha))**2 + 2 * g * h)) / g
    t_values = [t for t in [i * t_max / 1000 for i in range(1001)]]

    # Współrzędne toru lotu
    x = [V0 * math.cos(alpha) * t for t in t_values]
    y = [h + V0 * math.sin(alpha) * t - 0.5 * g * t**2 for t in t_values]

    # Rysowanie wykresu
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'b', label='Trajektoria')
    plt.title('Trajektoria lotu pocisku')
    plt.xlabel('Odległość [m]')
    plt.ylabel('Wysokość [m]')
    plt.grid(True)
    plt.savefig("trajektoria.png", dpi=300)
    plt.close()

target = random.randint(50, 340)
hit = False

while not hit:
    angle = input("Podaj kąt strzału: ")
    alpha = math.radians(int(angle))
    landing_spot = calculate_distance(alpha)
    print(target)
    print(landing_spot)
    if(abs(target-landing_spot)) < 5:
        print("trafiony")
        draw_plot(alpha)
        break





