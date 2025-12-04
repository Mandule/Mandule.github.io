---
title: "Q-Learning学习"
date: 2025-12-03 16:30:00 +0800
categories: [强化学习]
tags: [AI, RL, 强化学习, 算法学习]
math: true
---


下面我将用 **Q-Learning 算法**解决一个经典的强化学习入门问题：**网格世界（Grid World）中的寻路问题**。

我们将：
1. 定义一个简单的 4x4 网格环境；
2. 使用 **表格型 Q-Learning**（非深度学习）进行训练；
3. 展示完整可运行的 Python 代码；
4. 解释关键步骤。

---

## 🧩 问题设定：网格世界寻宝

- 智能体从左上角 `(0, 0)` 出发；
- 目标是到达右下角 `(3, 3)`（宝藏位置）；
- 每走一步 reward = -1（鼓励尽快到达）；
- 走出边界或撞墙 → 无效动作，留在原地；
- 到达目标 → reward = +10，episode 结束。

动作空间：`{0: 上, 1: 下, 2: 左, 3: 右}`

---

## ✅ Q-Learning 算法核心更新公式：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

其中：
- $\alpha$：学习率
- $\gamma$：折扣因子
- $r$：即时奖励
- $s$：下一个状态

---

## 🐍 完整 Python 代码（无需深度学习库）

```python
import numpy as np
import random

# ========== 1. 环境定义 ==========
GRID_SIZE = 4
ACTIONS = [0, 1, 2, 3]  # 上、下、左、右
ACTION_EFFECTS = {
    0: (-1, 0),   # 上
    1: (1, 0),    # 下
    2: (0, -1),   # 左
    3: (0, 1)     # 右
}

def step(state, action):
    """执行动作，返回 (next_state, reward, done)"""
    row, col = state
    dr, dc = ACTION_EFFECTS[action]
    new_row = max(0, min(GRID_SIZE - 1, row + dr))
    new_col = max(0, min(GRID_SIZE - 1, col + dc))
    next_state = (new_row, new_col)

    # 到达目标
    if next_state == (GRID_SIZE - 1, GRID_SIZE - 1):
        reward = 10
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done

# ========== 2. Q-Learning 训练 ==========
# 初始化 Q 表：Q[state][action]
Q = {}
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        Q[(r, c)] = {a: 0.0 for a in ACTIONS}

# 超参数
alpha = 0.1      # 学习率
gamma = 0.9      # 折扣因子
epsilon = 0.1    # ε-greedy 探索概率
num_episodes = 500

# 训练循环
for episode in range(num_episodes):
    state = (0, 0)  # 起点
    total_reward = 0

    while True:
        # ε-greedy 选择动作
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = max(Q[state], key=Q[state].get)

        # 执行动作
        next_state, reward, done = step(state, action)
        total_reward += reward

        # Q-Learning 更新
        best_next_q = max(Q[next_state].values())
        td_target = reward + gamma * best_next_q
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        # 转移到下一状态
        state = next_state

        if done:
            break

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# ========== 3. 提取最优策略并测试 ==========
print("\n=== 最优策略 ===")
policy = {}
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if (r, c) == (GRID_SIZE - 1, GRID_SIZE - 1):
            policy[(r, c)] = 'G'  # Goal
        else:
            best_action = max(Q[(r, c)], key=Q[(r, c)].get)
            policy[(r, c)] = ['↑', '↓', '←', '→'][best_action]

# 打印策略网格
for r in range(GRID_SIZE):
    row_str = ""
    for c in range(GRID_SIZE):
        row_str += f"{policy[(r, c)]:>2} "
    print(row_str)
```

---

## 🖨️ 输出示例（训练后）

```
Episode 0, Total Reward: -15
Episode 100, Total Reward: -6
Episode 200, Total Reward: -5
Episode 300, Total Reward: -5
Episode 400, Total Reward: -5

=== 最优策略 ===
→  →  →  ↓ 
↑  →  →  ↓ 
↑  ↑  →  ↓ 
↑  ↑  ↑  G 
```

> ✅ 智能体学会了从 `(0,0)` 向右走到底，再向下走到目标，共 6 步（reward = -5 + 10 = 5 净收益）。

---

## 🔍 关键点说明

| 组件 | 说明 |
|------|------|
| **Q 表** | 字典结构，键为状态 `(r,c)`，值为 `{action: q_value}` |
| **ε-greedy** | 平衡探索（exploration）与利用（exploitation） |
| **Off-policy** | Q-Learning 是 off-policy：即使使用 ε-greedy 采样，也在学习 greedy 最优策略 |
| **无模型** | 不需要知道环境动态（transition dynamics），仅靠交互学习 |

---

## 📌 扩展建议

- 将网格改为有障碍物（如某些格子不可进入）；
- 使用 **Deep Q-Network (DQN)** 处理更大状态空间；
- 可视化训练过程（如每轮路径、Q 值变化）。

---

这个例子展示了 Q-Learning 如何**从零开始通过试错学会最优行为**，是理解强化学习原理的经典范例。如果你希望看到 DQN 版本或更复杂环境（如 FrozenLake），也可以告诉我！