import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import fsolve

# Helper functions
def deg_to_rad(deg):
    return deg * np.pi / 180.0

def lines_of_bearing(x, y, angles, transmitters):
    # Перетворення кутів в радіани
    angles_rad = np.radians(angles)
    # Розрахунок ліній опори
    return [(y - ty) - np.tan(angle) * (x - tx) for (tx, ty), angle in zip(transmitters, angles_rad)]

def find_ship_position(transmitters, angles):
    #початкове припущення щодо положення корабля
    initial_guess = (0, 0)
    # функція, для якої потрібно знайти корінь
    def func_to_solve(coords):
        x, y = coords
        # Розрахунок
        bearings = lines_of_bearing(x, y, angles, transmitters)
        return bearings[:2]
    # Використовуєм fsolve, щоб знайти корінь системи рівнянь
    solution = fsolve(func_to_solve, initial_guess)
    return solution

# Подовжуєм лінії для візуалізації
def extend_line(x1, y1, angle, length=10):
    angle_rad = deg_to_rad(angle)
    x2 = x1 + length * np.cos(angle_rad)
    y2 = y1 + length * np.sin(angle_rad)
    return x1, y1, x2, y2

# дані (x, y, кут у градусах)
transmitters = [(8, 6), (-4, 5), (1, -3), (-1, -5), (0, 14), (4, 4)]
angles = [42, 158, 248, 260, 320, 100]

# Обчислення положення корабля
ship_position = find_ship_position(transmitters, angles)
print(f"Розрахункова позиція корабля: {ship_position}")

# Функція обчислення середньої позиції всіх перехресть
def calculate_average_position(intersections):
    # Обчислення середнього значення координат x і y
    average_x = np.mean([point[0] for point in intersections])
    average_y = np.mean([point[1] for point in intersections])
    return average_x, average_y

# Функція обчислення відхилення рішення
def calculate_deviation(intersections, average_position):
    #відстань від кожної точки перетину до середньої позиції
    distances = [np.sqrt((point[0] - average_position[0]) ** 2 + (point[1] - average_position[1]) ** 2) for point in intersections]
    #стандартне відхилення цих відстаней
    deviation = np.std(distances)
    return deviation

# Функція пошуку всіх перехресть
def find_all_intersections(transmitters, angles):
    intersections = []
    for i in range(len(transmitters)):
        for j in range(i + 1, len(transmitters)):
            solution = fsolve(lambda coords: lines_of_bearing(coords[0], coords[1], [angles[i], angles[j]], [transmitters[i], transmitters[j]]), (0, 0))
            intersections.append(solution)
    return intersections

# Знайти всі точки перетину
intersections = find_all_intersections(transmitters, angles)

#середнє положення всіх перехресть
average_position = calculate_average_position(intersections)
print(f"Середнє положення всіх перехресть: {average_position}")

# відхилення розв’язку
deviation = calculate_deviation(intersections, average_position)
print(f"Відхилення рішення: {deviation}")
# Plotting
fig, ax = plt.subplots()

# Ділянка передавачів і пеленгів
for idx, ((tx, ty), angle) in enumerate(zip(transmitters, angles)):
    x1, y1, x2, y2 = extend_line(tx, ty, angle)
    ax.plot([tx, x2], [ty, y2], 'k-')
    ax.plot(tx, ty, 'bo')
    ax.text(tx, ty, f' S{idx+1}', fontsize=9, verticalalignment='bottom')

    # Display angle
    arc_radius = 0.5
    arc = patches.Arc((tx, ty), arc_radius*2, arc_radius*2, theta1=0, theta2=angle, edgecolor='gray')
    ax.add_patch(arc)
    ax.text(tx + arc_radius / 1.5, ty, f'{angle}°', fontsize=8, color='gray')

# Проведення ліній x та y через (0, 0)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axvline(x=0, color='black', linewidth=0.8)

# Plot the estimated ship position
ax.plot(ship_position[0], ship_position[1], 'r*', markersize=10)

# Set plot limits
ax.axis('equal')
ax.set_xlim(min(transmitters, key=lambda t: t[0])[0] - 1, max(transmitters, key=lambda t: t[0])[0] + 1)
ax.set_ylim(min(transmitters, key=lambda t: t[1])[1] - 1, max(transmitters, key=lambda t: t[1])[1] + 1)


# Labels and title
plt.xlabel('X-кордината')
plt.ylabel('Y-кордината')
plt.title('Координати положення невідомого судна')
plt.grid(True)
plt.show()

# actual_position = (x_act, y_act) # кординати для окремої позиції

# Функція для обчислення відстані між двома точками
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Функція для малювання положень
def plot_positions(estimated_position, actual_position, transmitters, bearings):
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    # Малювання передавачів
    tx_x, tx_y = zip(*transmitters)
    plt.scatter(tx_x, tx_y, color='red', label='Передавачі')

    # Малювання теоретичного положення судна
    plt.scatter(*estimated_position, color='blue', label='Теоретичне положення судна')

    # Малювання фактичного положення судна
    plt.scatter(*actual_position, color='green', label='Фактичне положення судна')

    # Малювання ліній пеленгування
    for t, angle in zip(transmitters, bearings):
        x, y = t
        plt.plot([x, x + 10 * np.cos(np.radians(angle))], [y, y + 10 * np.sin(np.radians(angle))], 'k--')

    # Встановлення додаткових параметрів графіка
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Для забезпечення однакового масштабування обох осей
    plt.show()


# Тут ми вводимо фактичні координати

estimated_position = (2, 4)  # Припустимо, ці координати ми отримали з попередньої задачі
transmitters = [(0, 0), (5, 0), (5, 8)]  # Точки розташування передавачів
bearings = [30, 150, 270]  # Кути пеленгування в градусах

# Виконуємо обчислення
actual_position = (2, 5)  # Для прикладу, введемо припущення
distance = calculate_distance(estimated_position, actual_position)

# Виведення відстані
print(f"Відстань між теоретичним та фактичним положенням судна: {distance:.2f} одиниць")

# Викликаємо функцію малювання
plot_positions(estimated_position, actual_position, transmitters, bearings)

# Функція для визначення допустимої області навколо теоретичної позиції
def determine_allowable_region(transmitters, angles, theoretical_position, deviation):
    allowable_region = []

    for tx, ty in transmitters:
        # Обчислюємо різницю між теоретичною позицією та координатами передавача
        dx = tx - theoretical_position[0]
        dy = ty - theoretical_position[1]

        # Обчислюємо відстань від передавача до теоретичної позиції
        distance = np.sqrt(dx**2 + dy**2)

        # Якщо відстань менше за допустиме відхилення, то це допустимий регіон
        if distance <= deviation:
            allowable_region.append((tx, ty))

    return allowable_region

# Визначення допустимого відхилення (стандартне відхилення від розрахункового розв'язку)
deviation = calculate_deviation(intersections, average_position)

# Визначення допустимої області навколо теоретичної позиції судна
allowable_region = determine_allowable_region(transmitters, angles, ship_position, deviation)

# Перевірка, чи фактична позиція судна знаходиться в допустимій області
def check_actual_position(actual_position, allowable_region):
    for tx, ty in allowable_region:
        dx = tx - actual_position[0]
        dy = ty - actual_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        if distance <= deviation:
            return True
    return False

# Перевірка, чи фактична позиція судна знаходиться в допустимій області
is_in_allowable_region = check_actual_position(actual_position, allowable_region)

if is_in_allowable_region:
    print("Фактичне розташування судна знаходиться в допустимій області.")
else:
    print("Фактичне розташування судна не знаходиться в допустимій області.")

def select_optimal_transmitters(transmitters, angles, known_position):
    # Обчислення кутів пеленгування від кожного передавача до відомої точки
    bearings_to_known_position = [np.arctan2(known_position[1] - ty, known_position[0] - tx) for tx, ty in transmitters]

    # Обчислення різниці кутів
    angle_differences = [np.abs(a - b) for a, b in zip(bearings_to_known_position, np.radians(angles))]

    # Вибір передавачів з максимальною різницею кутів
    selected_indices = np.argsort(angle_differences)[-3:]  # 3 передавачі
    selected_transmitters = [transmitters[i] for i in selected_indices]
    selected_angles = [angles[i] for i in selected_indices]
    return selected_transmitters, selected_angles

# Застосування критерію
known_position = ship_position  # використовуємо визначену позицію як відому
optimal_transmitters, optimal_angles = select_optimal_transmitters(transmitters, angles, known_position)

# Виведення результатів у консоль
print("Оптимальні передавачі та їх кути пеленгування:")
for idx, (transmitter, angle) in enumerate(zip(optimal_transmitters, optimal_angles)):
    print(f"Передавач {idx+1}: координати {transmitter}, кут пеленгування {angle} градусів")

# Побудова графіка з оптимальними передавачами
plot_positions(ship_position, actual_position, optimal_transmitters, optimal_angles)


