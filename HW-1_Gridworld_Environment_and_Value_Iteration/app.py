from flask import Flask, render_template, request, jsonify
import random
import numpy as np

app = Flask(__name__)

# 初始化地圖
n = 5  # 預設大小
grid = [["" for _ in range(n)] for _ in range(n)]
start = None
end = None
obstacles = set()
actions = ["↑", "↓", "←", "→"]
policy = [[random.choice(actions) for _ in range(n)] for _ in range(n)]
value_function = np.zeros((n, n))

def evaluate_policy():
    """簡單的政策評估，計算 V(s)"""
    global value_function
    gamma = 0.9  # 折扣因子
    delta = 1e-3  # 收斂閾值

    while True:
        new_value_function = np.copy(value_function)
        for i in range(n):
            for j in range(n):
                if (i, j) in obstacles or (i, j) == start or (i, j) == end:
                    continue
                action = policy[i][j]
                ni, nj = i, j
                if action == "↑" and i > 0: ni -= 1
                elif action == "↓" and i < n-1: ni += 1
                elif action == "←" and j > 0: nj -= 1
                elif action == "→" and j < n-1: nj += 1
                new_value_function[i, j] = -1 + gamma * value_function[ni, nj]

        if np.max(np.abs(new_value_function - value_function)) < delta:
            break
        value_function = new_value_function

@app.route('/')
def index():
    return render_template("index.html", n=n, grid=grid, policy=policy, value_function=value_function.tolist())

@app.route('/set_size', methods=['POST'])
def set_size():
    global n, grid, policy, value_function, obstacles, start, end
    data = request.json
    n = data["size"]
    grid = [["" for _ in range(n)] for _ in range(n)]
    policy = [[random.choice(actions) for _ in range(n)] for _ in range(n)]
    value_function = np.zeros((n, n))
    obstacles.clear()
    start = None
    end = None
    return jsonify({
        "message": "Grid size updated", 
        "n": n, 
        "policy": policy, 
        "values": value_function.tolist()
    })

@app.route('/update_cell', methods=['POST'])
def update_cell():
    global start, end, obstacles
    data = request.json
    x, y = data["x"], data["y"]

    # 點擊設置起點、終點和障礙物
    if (x, y) == start:
        start = None  # 移除起點
        return jsonify({"message": "Start removed"})
    elif (x, y) == end:
        end = None  # 移除終點
        return jsonify({"message": "End removed"})
    elif (x, y) in obstacles:
        obstacles.remove((x, y))  # 移除障礙物
        return jsonify({"message": "Obstacle removed"})
    
    if not start:  # 如果沒有設置起點，設置為起點
        start = (x, y)
        return jsonify({"message": "Start set"})
    elif not end:  # 如果沒有設置終點，設置為終點
        end = (x, y)
        return jsonify({"message": "End set"})
    else:  # 若已經設置了起點和終點，設置為障礙物
        obstacles.add((x, y))
        return jsonify({"message": "Obstacle set"})

@app.route('/evaluate_policy', methods=['POST'])
def evaluate():
    evaluate_policy()
    return jsonify({"message": "Policy evaluated", "values": value_function.tolist()})

@app.route('/reset_values', methods=['POST'])
def reset_values():
    global value_function
    value_function = np.zeros((n, n))
    return jsonify({"message": "Values reset", "values": value_function.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
