from flask import Flask, request, render_template, jsonify
import plotly.graph_objs as go
import plotly 
import numpy as np
import json
from scipy.optimize import minimize
import time
from sympy import symbols, lambdify
from flask_cors import CORS
from pymongo import MongoClient
import html
import time 
import os

 
# Подключение к MongoDB с использованием указанного URL
client = MongoClient('mongodb+srv://usersdata:0hZzN9CAxapArcxI@cluster0.0fae0es.mongodb.net/mydatabase?retryWrites=true&w=majority&appName=Cluster0')
db = client.mydatabase  # Используем базу данных "mydatabase"


app = Flask(__name__)
CORS(app)  # Разрешает CORS для всех маршрутов и источников

CORS(app, resources={r"/*": {"origins": "*"}})  # Разрешить все источники
CORS(app, resources={r"/*": {"origins": ""}})  # Только для example.com
BASE_URL = os.getenv('BASE_URL', 'https://penalty-method-solution.onrender.com')  # Значение по умолчанию

# Определяем символические переменные для целевой функции
X1, X2 = symbols('x1 x2')

def objective_function(x, expression):
    x1, x2 = x
    # Создаем функцию из символьного выражения
    func = lambdify((X1, X2), expression, 'numpy')
    return -func(x1, x2)

def constraint1(x, constraint_expr, constraint_type):
    x1, x2 = x
    # print(len(constraint_type.replace(" ","")))
    # print("test", constraint_type.replace(" ","")==">")
    func = lambdify((X1, X2), constraint_expr, 'numpy')
    if constraint_type == ">=" or constraint_type.replace(" ","") == ">":
        return -func(x1, x2)
    elif constraint_type == "<=" or constraint_type.replace(" ","") == "<":
        return func(x1, x2)
    else:
        raise ValueError("Недопустимый тип ограничения")

def constraint2(x, constraint_expr, constraint_type):
    x1, x2 = x
    
    func = lambdify((X1, X2), constraint_expr, 'numpy')
    if constraint_type == ">=" or constraint_type.replace(" ","") == ">":
        return -func(x1, x2)
    elif constraint_type == "<=" or constraint_type.replace(" ","") == "<":
        return func(x1, x2)
    else:
        raise ValueError("Недопустимый тип ограничения")

    
def penalty_function_maximization(x1_min=-7, x1_max=7, x2_min=-7, x2_max=7, expression="(x1**2+x2**2-20*x1-30*x2)", 
                                  constraint_expr1="2*x1+3*x2-13", constraint_type1="<=", constraint_value1=0, 
                                  constraint_expr2="2*x1+x2-10", constraint_type2="<=", constraint_value2=0, 
                                  penalty_factor=1, reduction_factor=0.5, epsilon=0.00001, x1_initial=-8, x2_initial=-4):
    points = []
    penalty_values = []  # Для хранения значений penalty_factor
    penalty_factors = [] 
    epsilons = []


    constraint_value1 = float(constraint_value1)
    constraint_value2 = float(constraint_value2)

    points = [[x1_initial, x2_initial]]
    start_time = time.time()
    while True:
        def objective_function(x):
            return eval(expression, {'x1': x[0], 'x2': x[1]})

        def penalty(x):
            penalty1 = max(0, constraint1(x, constraint_expr1, constraint_type1) - constraint_value1) ** 2
            penalty2 = max(0, constraint2(x, constraint_expr2, constraint_type2) - constraint_value2) ** 2
            return penalty_factor * (penalty1 + penalty2)

        def total_objective(x):
            return objective_function(x) + penalty(x)

        result = minimize(total_objective, x0=[x1_initial, x2_initial], bounds=[(x1_min, x1_max), (x2_min, x2_max)])
        xk = result.x
        penalty_value = max(0, constraint1(xk, constraint_expr1, constraint_type1) - constraint_value1) ** 2 + max(0, constraint2(xk, constraint_expr2, constraint_type2) - constraint_value2) ** 2
        points.append(xk)
        penalty_values.append(penalty_value)  # Сохраняем текущее значение penalty_factor
        penalty_factors.append(penalty_factor)
        if penalty_value < epsilon:
            break

        penalty_factor /= reduction_factor

    return result, points, penalty_values, penalty_factors







def penalty_function_maximization_adaptive(epsilon, x1_min=-7, x1_max=7, x2_min=-7, x2_max=7, expression="(x1**2+x2**2-20*x1-30*x2)", 
                                           constraint_expr1="2*x1+3*x2-13", constraint_type1="<=", constraint_value1=0, 
                                           constraint_expr2="2*x1+x2-10", constraint_type2="<=", constraint_value2=0, 
                                           penalty_factor=1, increase_factor=1.1, x1_initial=-8, x2_initial=-4):
    points = []
    penalty_factors = []
    penalty_values = []  # Для хранения значений penalty_factor
    constraint_value1 = float(constraint_value1)
    constraint_value2 = float(constraint_value2)

    points = [[x1_initial, x2_initial]]
    start_time = time.time()
    while True:
        def objective_function(x):
            return eval(expression, {'x1': x[0], 'x2': x[1]})

        # Адаптивная штрафная функция
        def penalty(x):
            penalty1 = max(0, constraint1(x, constraint_expr1, constraint_type1) - constraint_value1) ** 2
            penalty2 = max(0, constraint2(x, constraint_expr2, constraint_type2) - constraint_value2) ** 2
            return penalty_factor * (penalty1 + penalty2)

        def total_objective(x):
            return objective_function(x) + penalty(x)

        result = minimize(total_objective, x0=[x1_initial, x2_initial], bounds=[(x1_min, x1_max), (x2_min, x2_max)])
        xk = result.x
        penalty_value = abs(penalty(xk))
        points.append(xk)
        penalty_factors.append(penalty_factor)
        penalty_values.append(penalty_value)
        
        if penalty_value < epsilon:
            break

        # Увеличение штрафного параметра
        penalty_factor *= increase_factor
        print(penalty_value)

    return result, points, penalty_values,penalty_factors





@app.route('/')
def index():
    expression = request.args.get('expression').replace('+', '%2B')
    constraint_expr1 = request.args.get('constraint_expr1').replace('+', '%2B')
    constraint_type1 = request.args.get('constraint_type1')
    constraint_value1 = float(request.args.get('constraint_value1'))
    constraint_expr2 = request.args.get('constraint_expr2').replace('+', '%2B')
    constraint_type2 = request.args.get('constraint_type2')
    constraint_value2 = float(request.args.get('constraint_value2'))
    function_type = request.args.get('function_type')
    epsilon = float(request.args.get("epsilon"))
    x1_initial = int(request.args.get("x1_initial"))
    x2_initial = int(request.args.get("x2_initial"))
    method = int(request.args.get("method"))
    points = request.args.get("points")



    return render_template('index.html', expression=expression, 
                           constraint_expr1=constraint_expr1,
                           constraint_type1=constraint_type1,
                           constraint_value1=constraint_value1,
                           constraint_expr2=constraint_expr2,
                           constraint_type2=constraint_type2, 
                           constraint_value2=constraint_value2, 
                           function_type=function_type,
                           epsilon=epsilon, 
                           x1_initial=x1_initial, 
                           x2_initial=x2_initial,
                           method=method,
                           points=points)



@app.route('/plot-data', methods=['GET'])
def plot_data():
    # Получаем параметры для задачи
    x1_min = -5.0
    x1_max = 5.0
    x2_min = -5.0
    x2_max = 5.0
    expression = request.args.get('expression')
    constraint_expr1 = request.args.get('constraint_expr1')
    constraint_type1 = request.args.get('constraint_type1')
    constraint_value1 = float(request.args.get('constraint_value1'))
    constraint_expr2 = request.args.get('constraint_expr2')
    constraint_type2 = request.args.get('constraint_type2')
    constraint_value2 = float(request.args.get('constraint_value2'))
    epsilon = float(request.args.get("epsilon"))
    method = int(request.args.get("method"))
    x1_initial = int(request.args.get("x1_initial"))
    x2_initial = int(request.args.get("x2_initial"))
    points = json.loads(html.unescape(request.args.get("points")))
    # print("fsdfdsfsd ", points)

    # Сохраняем результат и точки в базе данных
    points_data = points#[{'x1': p[0], 'x2': p[1]} for p in points]
    print(points_data)
    # Создаем лямбда-функции для ограничений
    constraint_func1 = lambdify((X1, X2), constraint_expr1, 'numpy')
    constraint_func2 = lambdify((X1, X2), constraint_expr2, 'numpy')

    # Подготовка данных для графика
    x = np.linspace(x1_min, x1_max, 100)
    y = np.linspace(x2_min, x2_max, 100)
    x, y = np.meshgrid(x, y)

    # Вычисляем целевую функцию
    z = eval(expression, {'x1': x, 'x2': y})

    # Применяем ограничения к каждой точке сетки
    constraint_check1 = constraint_func1(x, y)
    constraint_check2 = constraint_func2(x, y)

    if constraint_type1 == ">=":
        z[constraint_check1 < constraint_value1] = np.nan
    elif constraint_type1 == "<=":
        z[constraint_check1 > constraint_value1] = np.nan

    if constraint_type2 == ">=":
        z[constraint_check2 < constraint_value2] = np.nan
    elif constraint_type2 == "<=":
        z[constraint_check2 > constraint_value2] = np.nan

    # Создаем 3D график поверхности
    surface = go.Surface(z=z, x=x, y=y)

    # Разбиваем точки на координаты X, Y и Z
    points_x = [p['x1'] for p in points_data]
    points_y = [p['x2'] for p in points_data]
    points_z = [eval(expression, {'x1': p['x1'], 'x2': p['x2']}) for p in points_data]

    # Добавляем точки на график и соединяем их линией
    path = go.Scatter3d(
        x=points_x, 
        y=points_y, 
        z=points_z,
        mode='lines+markers',
        marker=dict(size=5, color='red'),
        line=dict(color='blue', width=2)
    )

    # Создаем фигуру с поверхностью и путем
    fig = go.Figure(data=[surface, path])

    # Настройка осей с равными масштабами
    fig.update_layout(
        title=f'График',
        autosize=True,
        width=600,
        height=600,
        margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="X1 Axis"),
            yaxis=dict(title="X2 Axis"),
            zaxis=dict(title="Objective Function")
        )
    )

    # Преобразуем график в JSON-формат для передачи на фронтенд
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # print("graphJSON", graphJSON)
    return jsonify(graphJSON)





@app.route('/solution-data', methods=['GET'])
def solution_data():
    expression = request.args.get('expression')
    constraint_expr1 = request.args.get('constraint_expr1')
    constraint_type1 = request.args.get('constraint_type1')
    constraint_value1 = float(request.args.get('constraint_value1'))
    constraint_expr2 = request.args.get('constraint_expr2')
    constraint_type2 = request.args.get('constraint_type2')
    constraint_value2 = float(request.args.get('constraint_value2'))
    epsilon = float(request.args.get("epsilon"))
    x1_initial = int(request.args.get("x1_initial"))
    x2_initial = int(request.args.get("x2_initial"))
    method = int(request.args.get('method'))
    print(method)



    # Начало измерения времени
    start_time = time.time()
    if method == 1:
        # Если данных нет, решаем задачу
        result, points, penalty_values, penalty_factors  = penalty_function_maximization(
            expression=expression,
            constraint_expr1=constraint_expr1,
            constraint_type1=constraint_type1,
            constraint_value1=constraint_value1,
            constraint_expr2=constraint_expr2,
            constraint_type2=constraint_type2,
            constraint_value2=constraint_value2,
            epsilon=epsilon, 
            x1_initial=x1_initial, 
            x2_initial=x2_initial
        )
    elif method == 3:
        result, points, penalty_values, penalty_factors  = penalty_function_maximization_adaptive(
            expression=expression,
            constraint_expr1=constraint_expr1,
            constraint_type1=constraint_type1,
            constraint_value1=constraint_value1,
            constraint_expr2=constraint_expr2,
            constraint_type2=constraint_type2,
            constraint_value2=constraint_value2,
            penalty_factor=1, 
            epsilon=epsilon, 
            x1_initial=x1_initial, 
            x2_initial=x2_initial
        )
    # Конец измерения времени
    end_time = time.time()
    # Вывод времени выполнения
    execution_time = end_time - start_time

    points_data = [{'x1': p[0], 'x2': p[1]} for p in points]
    solution_data = {
        'solution': {'x1': result.x[0], 'x2': result.x[1], 'fun': result.fun},
        'points': points_data,
        'penalty_values': penalty_values,  # Возвращаем список penalty_factors
        "penalty_factors" : penalty_factors,
        "execution_time" : execution_time
    }
    return jsonify(solution_data)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)  # Убедитесь, что хост — 0.0.0.0
