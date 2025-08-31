""""""
import copy
import math

import numpy as np
import pandas as pd

# 利用强化学习的一些概念，但是去除概率性的内容。
# 参考：https://blog.csdn.net/CltCj/article/details/119445005
# 状态 State
# 动作 Action
# 策略集（去概率） State.policies(self) -> list[Action]
# 状态转移（去概率） Action.transition(self, State) -> State
# 奖励 Action.reward : int
# 回报（return）表示未来的累积奖励
# 动作价值（action value）表示基于某策略pi在某状态s执行某动作a后，预期的回报
# 最优动作价值 Action.optimal_value -> int
# 最优状态价值 State.optimal_value -> int

# 从一个初始State开始，所有State和Action构成一棵树的节点和边，叶子节点为超时后的第一个状态

MAX_TIME = 150.0
HIGH_SPEED_TIME = 142.857
FULL_DISTANCE = 10000.0


class CharacterState:
    """角色状态"""
    basic_speed = 100
    max_energy = 100

    def __init__(self,
                 position: int,
                 speed: float,
                 energy: float = None,
                 distance: float = FULL_DISTANCE, ):
        if energy is None:
            energy = self.max_energy / 2.0
        self.position = position
        self.speed = speed
        self.energy = energy
        self.distance = distance

    @property
    def left_time(self):
        """计算剩余行动值"""
        return self.distance / self.speed

    def all_attack_action(self, parent):
        """获取普攻动作（遍历所有目标配置）"""
        return []

    def all_skill_action(self, parent):
        """获取战技动作（遍历所有目标配置）"""
        return []

    def all_ultimate_action(self, parent):
        """获取终结技动作（遍历所有目标配置）"""
        return []


class Firefly(CharacterState):
    basic_speed = 104
    max_energy = 240

    def all_attack_action(self, parent):
        a = CharacterAction(parent, position=self.position, target=[], reward=1, name=f'{self.position}')

        def modify(new):
            new.characters[self.position].distance = FULL_DISTANCE

        a.modify = modify
        return [a]


class Bronya(CharacterState):
    basic_speed = 99
    max_energy = 120

    def all_skill_action(self, parent):
        actions = []
        for target in [0]:
            a = CharacterAction(parent, position=self.position, target=[target], reward=0, name=f'{self.position}')

            def modify(new):
                new.characters[self.position].distance = FULL_DISTANCE
                new.characters[target].distance = 0.0

            a.modify = modify
            actions.append(a)
        return actions

    def all_ultimate_action(self, parent):
        a = CharacterAction(parent, position=self.position, target=[], reward=0, name=f'U{self.position}')

        def modify(new):
            for c in new.characters:
                c.distance = max(0.0, c.distance - 2500.0)
            new.characters[self.position].energy = 0.0

        a.modify = modify
        return [a]


class State:
    """状态"""

    def __init__(self,
                 sequence: list[CharacterState],
                 time: float = 0.0,
                 parent=None,
                 modify=None):
        self.parent: Action | None = parent  # 父动作
        self.characters = sorted(sequence, key=lambda s: s.position)
        self.time = time
        if modify is not None:
            modify(self)
        self.sequence = sorted(sequence, key=CharacterState.left_time.fget)  # 行动序列
        self._children = None
        self._optimal_child = None

    @property
    def children(self):
        """状态的所有可能动作"""
        if self._children is None:
            self._children: list[Action] = self._policies()
        return self._children

    @property
    def optimal_child(self):
        """状态的最优动作"""
        if self._optimal_child is None:
            self._optimal_child: Action | None = max(self.children, key=Action.optimal_value.fget, default=None)
        return self._optimal_child

    @property
    def optimal_value(self):
        """状态的最优价值"""
        if self.optimal_child is None:
            return 0
        return self.optimal_child.optimal_value

    def _policies(self):
        """策略集（获取状态的所有可能动作）"""
        actions: list[Action] = []

        # 若此状态已超时，作为叶子节点返回空列表
        if self.time > MAX_TIME:
            return []
        # 遍历每个位置是否有终结技
        for c in self.characters:
            if c.energy >= c.max_energy:
                actions += c.all_ultimate_action(self)
        # 遍历每个位置是否行动（按照行动序列）
        for c in self.sequence:
            if c.distance < 0.01:
                actions += c.all_attack_action(self)
                actions += c.all_skill_action(self)
                break
        else:
            # 无人行动时推进时间
            a = TimePassAction(self, name='TP')
            actions.append(a)

        return actions

    def copy(self, parent, modify):
        """基于当前状态，创建一个新的状态，并指定其父动作"""
        return State(sequence=copy.deepcopy(self.sequence),
                     time=self.time,
                     modify=modify,
                     parent=parent)


class Action:
    """动作"""

    def __init__(self,
                 parent: State,
                 reward: int = 0,
                 name: str = None):
        if name is None:
            name = self.__class__
        self.name = name
        self.reward = reward
        self.parent: State = parent  # 父状态
        self._child = None
        self._optimal_value = None

    @property
    def child(self):
        """动作后的状态"""
        if self._child is None:
            self._child: State = self._transition()
        return self._child

    @property
    def optimal_value(self):
        """动作的最优价值"""
        if self._optimal_value is None:
            self._optimal_value: int = self.reward + self.child.optimal_value
        return self._optimal_value

    def __str__(self):
        return f"({self.name}, time: {self.parent.time:.2f})"

    def _transition(self) -> State:
        """状态转移（获取动作后的状态）"""
        return self.parent.copy(parent=self, modify=self.modify)

    def modify(self, new: State):
        """动作对状态的修改"""
        pass


class TimePassAction(Action):
    """时间推进至下一个状态"""

    def modify(self, new: State):
        p = self.parent
        passed_time = p.sequence[0].left_time
        if p.time < HIGH_SPEED_TIME <= p.time + passed_time:
            passed_time = HIGH_SPEED_TIME - p.time
            new.characters[0].speed -= 60
        new.time += passed_time
        for c, new_c in zip(p.characters, new.characters):
            new_c.distance -= passed_time * c.speed


class CharacterAction(Action):
    """角色常规动作（普攻/战技）"""

    def __init__(self,
                 parent: State,
                 position: int,
                 target: list[int] = None,
                 reward: int = 0,
                 name: str = ''):
        if target is None:
            target = []
        self.position = position
        self.target = target
        super().__init__(parent, reward, name)


"""单点情况追踪"""
c0 = Firefly(0, speed=250, energy=0.0)
c1 = Bronya(1, speed=139, energy=120.0)
init_state = State(sequence=[c0, c1])
print(init_state.optimal_value)
s = init_state
while True:
    a = s.optimal_child
    if a is None:
        break
    print(a)
    s = a.child

"""扫描参数空间"""
# 流萤基础速度：104
# 布洛妮娅基础速度：99
basic_speed = [104, 99]
# 固定加速（行迹属性、非遗器效果）
fixed_speed_add = [5 + 60, 0]
fixed_speed_boost = [0.1, 0.1]
# 遗器加速理论范围
min_speed_add = 0.0
min_speed_boost = -0.08
max_speed_add = 25 + 2.6 * 30
max_speed_boost = 0.06 * 3
# 计算速度范围
min_speed = [math.ceil(bs * (1 + fsb + min_speed_boost) + fsa + min_speed_add)
             for bs, fsa, fsb in zip(basic_speed, fixed_speed_add, fixed_speed_boost)]
max_speed = [math.floor(bs * (1 + fsb + max_speed_boost) + fsa + max_speed_add)
             for bs, fsa, fsb in zip(basic_speed, fixed_speed_add, fixed_speed_boost)]
speed_range = [np.arange(a, b + 1) for a, b in zip(min_speed, max_speed)]

# 遍历所有速度配置
round_table = np.zeros([len(sr) for sr in speed_range])

for s0 in range(min_speed[0], max_speed[0] + 1):
    for s1 in range(min_speed[1], max_speed[1] + 1):
        c0 = Firefly(0, speed=s0, energy=0.0)
        c1 = Bronya(1, speed=s1, energy=120.0)
        init_state = State(sequence=[c0, c1])
        t = init_state.optimal_value
        round_table[s0 - min_speed[0]][s1 - min_speed[1]] = t

# 保存数据
df = pd.DataFrame(round_table.T, index=speed_range[1], columns=speed_range[0])
csv_filename = "round_table_data.csv"
df.to_csv(csv_filename)
print(f"Round table saved to {csv_filename}")
