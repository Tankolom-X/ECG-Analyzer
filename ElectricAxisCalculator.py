import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def calculate_electrical_axis(Q1, R1, S1, Q3, R3, S3):
    """
    Рассчитывает среднюю электрическую ось сердца по методу треугольника Эйнтховена

    Параметры:
    Q1, R1, S1 - амплитуды зубцов в I отведении (мВ)
    Q3, R3, S3 - амплитуды зубцов в III отведении (мВ)

    Возвращает:
    angle - угол альфа между горизонталью и электрической осью сердца (градусы)
    """
    # Рассчитываем алгебраическую сумму для I и III отведений
    sum_I = round(R1 - (abs(Q1) + abs(S1)), 4)
    sum_III = round(R3 - (abs(Q3) + abs(S3)), 4)

    # Координаты вершин треугольника Эйнтховена (перевернутый треугольник)
    A = np.array([-1, 0])  # Правая рука
    B = np.array([1, 0])  # Левая рука
    C = np.array([0, -np.sqrt(3)])  # Левая нога

    # Центр треугольника
    O = np.array([0, -np.sqrt(3) / 3])

    # Точки проекции центра на стороны AB (I отведение) и BC (III отведение)
    O1 = np.array([0, 0])  # Середина AB
    O3 = np.array([0.5, -np.sqrt(3) / 2])  # Середина BC

    # Откладываем отрезки на сторонах AB и BC
    scale_factor = 0.2  # Масштабный коэффициент

    # Для I отведения (AB)
    X1 = O1 + np.array([sum_I * scale_factor, 0])

    # Для III отведения (BC)
    BC_vector = C - B
    BC_unit_vector = BC_vector / np.linalg.norm(BC_vector)
    X3 = O3 + sum_III * scale_factor * BC_unit_vector

    # Перпендикуляры к сторонам через X1 и X3 (внутрь треугольника)
    perp_X1 = np.array([X1[0], X1[1] - 2])  # Перпендикуляр к AB через X1 (вниз)

    # Перпендикуляр к BC через X3 (направлен внутрь треугольника)
    perp_X3_dir = np.array([BC_vector[1], -BC_vector[0]])
    perp_X3_dir = perp_X3_dir / np.linalg.norm(perp_X3_dir)
    perp_X3 = X3 + perp_X3_dir * 2

    # Находим точку пересечения перпендикуляров (K)
    m = perp_X3_dir[1] / perp_X3_dir[0]
    x_k = X1[0]
    y_k = X3[1] + m * (x_k - X3[0])
    K = np.array([x_k, y_k])

    # Вектор электрической оси
    OK_vector = K - O

    # Упрощенный расчет угла между горизонталью и OK
    angle_deg = np.degrees(np.arctan2(OK_vector[1], OK_vector[0]))

    # Нормализуем угол
    angle_deg = angle_deg % 360
    if angle_deg > 180:
        angle_deg -= 360
    angle_deg = -angle_deg

    return (angle_deg, O, K, A, B, C, X1, X3, perp_X1, perp_X3, O1, O3,
            sum_I, sum_III)


def plot_einthoven_triangle(angle, O, K, A, B, C, X1, X3, perp_X1, perp_X3, O1, O3,
                            sum_I, sum_III, Q1, R1, S1, Q3, R3, S3):
    """Визуализация треугольника Эйнтховена и электрической оси"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Треугольник Эйнтховена
    triangle = Polygon([A, B, C], closed=True, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(triangle)

    # Вершины треугольника
    ax.text(A[0] - 0.1, A[1] + 0.1, 'A (Правая рука)', ha='right', va='bottom')
    ax.text(B[0] + 0.1, B[1] + 0.1, 'B (Левая рука)', ha='left', va='bottom')
    ax.text(C[0], C[1] - 0.15, 'C (Левая нога)', ha='center', va='top')

    # Центр и проекции
    ax.plot(O[0], O[1], 'ro', markersize=5)
    ax.text(O[0], O[1] + 0.1, 'O (Центр)', ha='center', va='bottom')

    # Точки и отрезки
    ax.plot(O1[0], O1[1], 'bo', markersize=4)
    ax.text(O1[0], O1[1] + 0.05, 'O1', ha='center', va='bottom', color='b')
    ax.plot(O3[0], O3[1], 'bo', markersize=4)
    ax.text(O3[0], O3[1] + 0.05, 'O3', ha='center', va='bottom', color='b')

    ax.plot(X1[0], X1[1], 'go', markersize=4)
    ax.text(X1[0], X1[1] + 0.05, 'X1', ha='center', va='bottom', color='g')
    ax.plot(X3[0], X3[1], 'go', markersize=4)
    ax.text(X3[0], X3[1] + 0.05, 'X3', ha='center', va='bottom', color='g')

    # Отрезки на сторонах
    ax.plot([O1[0], X1[0]], [O1[1], X1[1]], 'g-', linewidth=2)
    ax.plot([O3[0], X3[0]], [O3[1], X3[1]], 'g-', linewidth=2)

    # Дополнительные линии (O3-A и O1-C)
    ax.plot([O3[0], A[0]], [O3[1], A[1]], '--', color='gray', alpha=0.5)
    ax.plot([O1[0], C[0]], [O1[1], C[1]], '--', color='gray', alpha=0.5)

    # Перпендикуляры
    ax.plot([X1[0], perp_X1[0]], [X1[1], perp_X1[1]], 'r--')
    ax.plot([X3[0], perp_X3[0]], [X3[1], perp_X3[1]], 'r--')

    # Горизонтальная линия через O для отсчета угла
    ax.plot([-1, 1], [O[1], O[1]], 'b--', alpha=0.5)

    # Электрическая ось
    ax.plot(K[0], K[1], 'ro', markersize=5)
    ax.text(K[0], K[1] - 0.05, 'K', ha='center', va='top')
    ax.plot([O[0], K[0]], [O[1], K[1]], 'm-', linewidth=2)
    ax.arrow(O[0], O[1], (K[0] - O[0]) * 0.9, (K[1] - O[1]) * 0.9,
             head_width=0.05, head_length=0.1, fc='m', ec='m')

    # Угол
    angle_text = f'Угол α = {angle:.1f}°'

    ax.text(0.5, 0.2, angle_text, ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8), fontsize=12)

    # Настройки графика
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 0.5)
    ax.set_aspect('equal')
    ax.set_title('Треугольник Эйнтховена с электрической осью сердца', pad=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Программа для расчета электрической оси сердца")
    print("Введите амплитуды зубцов (в мВ):")

    Q1 = float(input("Q в I отведении (мВ): ").replace(',', '.'))
    R1 = float(input("R в I отведении (мВ): ").replace(',', '.'))
    S1 = float(input("S в I отведении (мВ): ").replace(',', '.'))

    Q3 = float(input("Q в III отведении (мВ): ").replace(',', '.'))
    R3 = float(input("R в III отведении (мВ): ").replace(',', '.'))
    S3 = float(input("S в III отведении (мВ): ").replace(',', '.'))

    # Расчет электрической оси
    angle, O, K, A, B, C, X1, X3, perp_X1, perp_X3, O1, O3, sum_I, sum_III = calculate_electrical_axis(
        Q1, R1, S1, Q3, R3, S3)

    print("\nРезультаты:")
    print(f"Сумма I: {sum_I:.4f} мВ, Сумма III: {sum_III:.4f} мВ")
    print(f"Угол α: {angle:.1f}°")


    # Построение треугольника
    plot_einthoven_triangle(angle, O, K, A, B, C, X1, X3, perp_X1, perp_X3, O1, O3,
                            sum_I, sum_III, Q1, R1, S1, Q3, R3, S3)