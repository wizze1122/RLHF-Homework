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

def value_iteration():
    """價值迭代算法，計算最佳政策"""
    global value_function, policy
    gamma = 0.9  # 折扣因子
    delta = 1e-3  # 收斂閾值
    
    # 初始化價值函數
    value_function = np.zeros((n, n))
    
    # 設置終點價值
    if end:
        value_function[end[0], end[1]] = 0
    
    # 障礙物的處理方式修改：不顯示為無限值
    for i in range(n):
        for j in range(n):
            if (i, j) in obstacles:
                value_function[i, j] = -100  # 使用很低的值而非無限
    
    while True:
        delta_value = 0
        new_value_function = np.copy(value_function)
        
        for i in range(n):
            for j in range(n):
                if (i, j) in obstacles or (i, j) == end:
                    continue
                
                # 計算每個動作的價值
                action_values = []
                
                # 向上
                if i > 0 and (i-1, j) not in obstacles:
                    action_values.append((-1 + gamma * value_function[i-1, j], "↑"))
                else:
                    action_values.append((-1 + gamma * value_function[i, j], "↑"))
                
                # 向下
                if i < n-1 and (i+1, j) not in obstacles:
                    action_values.append((-1 + gamma * value_function[i+1, j], "↓"))
                else:
                    action_values.append((-1 + gamma * value_function[i, j], "↓"))
                
                # 向左
                if j > 0 and (i, j-1) not in obstacles:
                    action_values.append((-1 + gamma * value_function[i, j-1], "←"))
                else:
                    action_values.append((-1 + gamma * value_function[i, j], "←"))
                
                # 向右
                if j < n-1 and (i, j+1) not in obstacles:
                    action_values.append((-1 + gamma * value_function[i, j+1], "→"))
                else:
                    action_values.append((-1 + gamma * value_function[i, j], "→"))
                
                # 找出最佳動作及其價值
                best_value, best_action = max(action_values)
                
                # 更新價值函數和政策
                new_value_function[i, j] = best_value
                policy[i][j] = best_action
                
                # 計算變化量
                delta_value = max(delta_value, abs(new_value_function[i, j] - value_function[i, j]))
        
        # 更新價值函數
        value_function = new_value_function
        
        # 檢查是否收斂
        if delta_value < delta:
            break

def find_optimal_path():
    """基於當前政策找出從起點到終點的最佳路徑"""
    if not start or not end:
        return []
    
    path = [start]
    current = start
    
    # 防止無限循環
    max_steps = n * n
    step_count = 0
    
    while current != end and step_count < max_steps:
        i, j = current
        action = policy[i][j]
        
        # 根據政策確定下一步
        ni, nj = i, j
        if action == "↑" and i > 0: ni -= 1
        elif action == "↓" and i < n-1: ni += 1
        elif action == "←" and j > 0: nj -= 1
        elif action == "→" and j < n-1: nj += 1
        
        # 如果下一步是障礙物或走出邊界，結束
        if (ni, nj) in obstacles or ni < 0 or ni >= n or nj < 0 or nj >= n:
            break
        
        current = (ni, nj)
        path.append(current)
        step_count += 1
        
        # 如果回到已經訪問過的狀態，表示存在循環，則跳出
        if path.count(current) > 1:
            break
    
    # 檢查是否達到終點
    if current != end:
        return []  # 找不到有效路徑
    
    return path

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
    message = ""
    do_not_iterate = data.get("doNotIterate", False)

    # 點擊設置起點、終點和障礙物
    if (x, y) == start:
        start = None  # 移除起點
        message = "起點已移除"
    elif (x, y) == end:
        end = None  # 移除終點
        message = "終點已移除"
    elif (x, y) in obstacles:
        obstacles.remove((x, y))  # 移除障礙物
        message = "障礙物已移除"
    elif not start:  # 設置起點
        start = (x, y)
        message = "起點已設置"
    elif not end:  # 設置終點
        end = (x, y)
        message = "終點已設置"
        # 當終點被設置後，如果不是禁止迭代的請求，自動執行價值迭代
        if start and end and not do_not_iterate:
            value_iteration()
            path = find_optimal_path()
            return jsonify({
                "message": message,
                "runIteration": True,
                "values": value_function.tolist(),
                "policy": policy,
                "path": path
            })
    else:  # 設置障礙物
        obstacles.add((x, y))
        message = "障礙物已設置"
        # 如果起點和終點都已設置，且不是禁止迭代的請求，自動執行價值迭代
        if start and end and not do_not_iterate:
            value_iteration()
            path = find_optimal_path()
            return jsonify({
                "message": message,
                "runIteration": True,
                "values": value_function.tolist(),
                "policy": policy,
                "path": path
            })
    
    return jsonify({"message": message, "runIteration": False})

@app.route('/evaluate_policy', methods=['POST'])
def evaluate():
    evaluate_policy()
    return jsonify({"message": "政策評估完成", "values": value_function.tolist()})

@app.route('/value_iteration', methods=['POST'])
def run_value_iteration():
    value_iteration()
    path = find_optimal_path()
    return jsonify({
        "message": "價值迭代完成",
        "values": value_function.tolist(),
        "policy": policy,
        "path": path
    })

@app.route('/reset_values', methods=['POST'])
def reset_values():
    global value_function
    value_function = np.zeros((n, n))
    return jsonify({"message": "Values reset", "values": value_function.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
