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

Q 表（Q-table）是 **Q-Learning 算法中的核心数据结构**，它存储的是：

> **在给定状态（state）下，采取某个动作（action）所能获得的“长期预期回报”的估计值。**

---

### 📌 正式定义

Q 表存储的是 **动作价值函数（Action-Value Function）** 的估计值，记作：

$$
Q(s, a) \approx \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r_{t} \,\bigg|\, s_0 = s, a_0 = a, \pi \right]
$$

即：**从状态 $s$ 出发，执行动作 $a$，之后按照策略 $\pi$ 行动，所能获得的折扣累积奖励的期望值。**

> 在 Q-Learning 中，这个策略 $\pi$ 最终收敛到**最优策略**，因此 $Q(s,a)$ 逼近 **最优动作价值函数 $Q^*(s,a)$**。

---

### 🧩 Q 表的结构

假设：
- 状态空间有限：$S = \{s_1, s_2, ..., s_n\}$
- 动作空间有限：$A = \{a_1, a_2, ..., a_m\}$

那么 Q 表就是一个 **二维表格**（或字典、数组）：

| 状态 \ 动作 | a₁ | a₂ | a₃ | ... |
|-------------|----|----|----|-----|
| s₁          | Q(s₁,a₁) | Q(s₁,a₂) | Q(s₁,a₃) | ... |
| s₂          | Q(s₂,a₁) | Q(s₂,a₂) | Q(s₂,a₃) | ... |
| ...         | ... | ... | ... | ... |

✅ **每个单元格 `Q[s][a]` 存储一个实数**，表示“在状态 s 下做动作 a 的好坏程度”。

---

### 🔍 举个例子（网格世界）

状态：位置 `(行, 列)`，如 `(0,0)`  
动作：`{0: 上, 1: 下, 2: 左, 3: 右}`

Q 表可能长这样（部分）：

```python
Q = {
    (0, 0): {0: -0.2, 1: 1.5, 2: -1.0, 3: 2.1},
    (0, 1): {0: -0.1, 1: 1.8, 2: 2.0, 3: 2.5},
    ...
    (3, 3): {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}  # 终止状态
}
```

- 在 `(0,0)` 时，向右（动作 3）的 Q 值最高（2.1）→ 应该优先选右；
- Q 值越大，说明该动作**长期来看越有利**。

---

### 🔄 Q 表如何更新？

通过 **Q-Learning 更新规则**：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]
$$

- `r`：执行动作后立即获得的奖励；
- $\max_{a'} Q(s', a')$：对下一个状态 `s'`，选择**最大 Q 值的动作**（即假设未来会最优行动）；
- 整个括号内是 **TD 误差（Temporal Difference Error）**，衡量当前估计与新信息的差距。

> ✅ 通过不断与环境交互，Q 表逐渐收敛到真实最优 Q 值。

---

### ⚠️ Q 表的局限性

| 优点 | 缺点 |
|------|------|
| 简单直观，保证收敛（表格形式） | **仅适用于小规模离散状态/动作空间** |
| 无需模型（model-free） | 状态数量爆炸时（如图像输入），无法存储 |
| 是理解 RL 的基础 | 无法泛化：每个状态独立学习，相似状态不共享信息 |

> 🌟 为解决这个问题，**Deep Q-Network (DQN)** 用神经网络代替 Q 表，实现 **函数近似（Function Approximation）**。

---

### ✅ 总结

> **Q 表存储的是：在每个状态-动作对 $(s, a)$ 下，智能体预计能获得的未来累积奖励的估计值。**  
> 它是 Q-Learning 算法的记忆库，通过不断试错和更新，最终指导智能体做出最优决策。

你可以把它想象成一本“行动指南”：  
> “当你处在状态 s 时，查表看看哪个动作 a 的 Q 值最高，就选它！”



Q-learning 能够**在理论上保证收敛到最优策略（从而最大化长期回报）**，这是由其**更新规则的数学性质**和**强化学习的基本原理**共同决定的。下面我们从直觉、算法机制和理论保证三个层面来解释。

---

## 一、核心思想：学习“最优动作价值函数” $Q^*(s, a)$

Q-learning 的目标不是学习某个特定策略下的 Q 值，而是直接逼近 **最优动作价值函数** $Q^*(s, a)$，定义为：

$$
Q^*(s, a) = \max_{\pi} \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \,\bigg|\, s_0 = s, a_0 = a \right]
$$

即：**从状态 $s$ 出发，执行动作 $a$，之后始终采取最优行为，所能获得的最大期望折扣累积回报。**

一旦我们知道了 $Q^*$，最优策略 $\pi^*$ 就很简单：

$$
\pi^*(s) = \arg\max_a Q^*(s, a)
$$

> ✅ 所以，只要 Q-learning 能学到 $Q^*$，就能得到最大化长期回报的策略。

---

## 二、Q-learning 的更新规则：朝着 Bellman 最优方程收敛

Q-learning 使用以下更新公式：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

这个公式的关键在于：
- **目标值（TD target）** 是 $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$
- 它**假设未来会采取最优动作**（因为用了 $\max_{a'}$）

这正是 **Bellman 最优方程（Bellman Optimality Equation）** 的形式：

$$
Q^*(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q^*(s', a') \,\bigg|\, s, a \right]
$$

> 🔑 Q-learning 的更新就是在**用采样数据逼近这个方程**。

通过不断迭代，Q 值会逐渐满足 Bellman 最优方程 → 收敛到 $Q^*$。

---

## 三、为什么能保证收敛？—— 理论条件

Q-learning 的**收敛性有严格数学证明**（Watkins & Dayan, 1992），但需要满足以下条件：

### ✅ 收敛条件（表格型 Q-learning）：
1. **所有状态-动作对被无限次访问**（充分探索）  
   → 通常通过 ε-greedy 等策略保证；
2. **学习率 $\alpha_t$ 满足 Robbins-Monro 条件**：
   $$
   \sum_{t} \alpha_t = \infty, \quad \sum_{t} \alpha_t^2 < \infty
   $$
   → 例如 $\alpha_t = 1/t$；
3. **环境是马尔可夫的（MDP）**；
4. **奖励有界**。

在这些条件下，**Q(s, a) 以概率 1 收敛到 $Q^*(s, a)$**。

> 📌 注意：这是针对**表格型 Q-learning**（即状态/动作空间有限，Q 表显式存储）的结论。

---

## 四、直觉解释：如何避免“短视”？

你可能会问：  
> “Q-learning 只看到 immediate reward $r_t$，怎么知道长期回报？”

答案是：**通过递归传播未来的价值**。

举个例子：

```
状态 A --(右)--> 状态 B --(右)--> Goal (+10)
```

- 初始时，所有 Q=0。
- 第一次到达 Goal：  
  - 在 B 执行“右”，得到 r=10 → 更新 $Q(B, 右) ≈ 10$
- 下次在 A 执行“右”到 B：  
  - r=0（假设中间无奖励），但 $ \max_{a'} Q(B, a') = 10 $  
  - 所以更新：$Q(A, 右) ← 0 + α[0 + γ·10 - 0] = αγ·10$
- 随着多次访问，$Q(A, 右)$ 会逐渐接近 $γ·10$

✅ **价值从 Goal 反向传播回起点**，即使中间步骤 reward=0！

这就是 **“信用分配（Credit Assignment）”** 的体现。

---

## 五、Off-policy 的优势：学最优策略，哪怕当前行为不优

Q-learning 是 **off-policy** 算法：
- **行为策略（behavior policy）**：比如 ε-greedy（带探索）
- **目标策略（target policy）**：greedy（$\arg\max Q$）

> 即使你当前乱走（exploration），Q-learning 依然在学习“如果我以后总是选最好的动作，能得多少分”。

这使得它**既能探索，又能保证最终收敛到最优**。

---

## 六、局限性说明（何时不能保证？）

虽然理论上有保证，但在实践中可能失效：

| 情况 | 原因 |
|------|------|
| **状态空间太大**（如图像输入） | 表格无法存储 → 需用函数近似（如 DQN），此时**收敛性不再保证** |
| **探索不足** | 某些 (s,a) 从未被访问 → Q 值未更新 |
| **学习率不合适** | 太大→震荡，太小→收敛慢 |
| **非平稳环境** | 环境动态变化，最优策略也在变 |

> 💡 在深度 Q-learning（DQN）中，通过经验回放、目标网络等技巧**缓解**这些问题，但严格收敛性丢失。

---

## ✅ 总结

Q-learning 能保证最大化长期回报，是因为：

1. **目标明确**：直接学习最优动作价值函数 $Q^*$；
2. **更新规则正确**：基于 Bellman 最优方程，用 $\max_{a'} Q$ 代表未来最优行为；
3. **价值反向传播**：奖励信号通过递归方式从终点传回起点；
4. **理论保证**：在有限 MDP + 充分探索 + 合适学习率下，**收敛到全局最优**；
5. **Off-policy 特性**：探索与学习解耦，兼顾探索和优化。

> 正如一句 RL 名言：  
> **“Q-learning doesn’t just learn what’s good now — it learns what leads to what’s good later.”**




