""""""
import copy
import math
import time

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

# 系统结构
# 状态 State
# . 角色状态 CharacterState
#   . Buff
# . 倒计时 Timer
# -> 动作 Action
# 动作 Action
# -> 状态 State

MAX_TIME = 150.0
HIGH_SPEED_TIME = 142.857
FULL_DISTANCE = 10000.0


class Buff:
    """Buff"""
    name = '未命名buff'  # 名称

    def __init__(self,
                 turn: int | None = None,  # 剩余回合数（None为永续）
                 stack: int | None = None,  # 层数（None为不可叠层）
                 ):
        self.turn = turn  # 剩余回合数
        self.stack = stack  # 层数

    def enable(self, character):
        """添加buff"""
        pass

    def disable(self, character):
        """删除buff"""
        pass


class SomatotypicalHelix(Buff):
    name = "分型的螺旋"

    def enable(self, character):
        character.speed += 0.098 * character.base_speed
        character.buffs.append(self)


class CompleteCombustion(Buff):
    name = "完全燃烧"

    def enable(self, character):
        character.speed += 60
        character.buffs.append(self)

    def disable(self, character):
        character.speed -= 60
        character.buffs.remove(self)


class CharacterState:
    """角色状态"""
    # 以下是战斗中不变的属性
    name = '未命名角色'  # 名称
    # combat_type = 1  # 战斗属性（1-7：物火冰雷风量虚）
    # path = 1  # 命途（1-8：毁灭 巡猎 智识 同协 虚无 存护 丰饶 记忆）
    # level = 80  # 等级
    # base_max_hp = 2000  # 基础生命上限
    # base_attack = 1000  # 基础攻击力
    # base_defence = 1000  # 基础防御力
    base_speed = 100  # 基础速度
    max_energy = 100  # 能量上限

    def __init__(self,
                 position: int,  # 编队位置
                 # max_hp: float,  # 生命上限
                 # hp: float,  # 生命值
                 # attack: float,  # 攻击力
                 # defence: float,  # 防御力
                 speed: float,  # 速度
                 distance: float,  # 距离
                 energy: float,  # 能量
                 energy_regeneration_rate: float,  # 能量恢复效率
                 # crit_rate: float,  # 暴击率
                 # crit_damage: float,  # 暴击伤害
                 # break_effect: float,  # 击破特攻
                 # outgoing_healing_boost: float,  # 治疗量加成
                 # effect_hit_rate: float,  # 效果命中
                 # effect_resistance: float,  # 效果抵抗
                 # damage_boost: list[float],  # 伤害提高
                 # resistance_boost: list[float],  # 抗性提高
                 buffs: list[Buff]  # 增益/减益效果
                 ):
        self.position = position  # 编队位置
        # self.max_hp = max_hp  # 生命上限
        # self.hp = hp  # 生命值
        # self.attack = attack  # 攻击力
        # self.defence = defence  # 防御力
        self.speed = speed  # 速度
        self.distance = distance  # 距离
        self.energy = energy  # 能量
        self.energy_regeneration_rate = energy_regeneration_rate  # 能量恢复效率
        # self.crit_rate = crit_rate  # 暴击率
        # self.crit_damage = crit_damage  # 暴击伤害
        # self.break_effect = break_effect  # 击破特攻
        # self.outgoing_healing_boost = outgoing_healing_boost  # 治疗量加成
        # self.effect_hit_rate = effect_hit_rate  # 效果命中
        # self.effect_resistance = effect_resistance  # 效果抵抗
        # self.damage_boost = damage_boost  # 伤害提高
        # self.resistance_boost = resistance_boost  # 抗性提高
        self.buffs = buffs  # 增益/减益效果

    @property
    def action_value(self):
        """计算剩余行动值"""
        return self.distance / self.speed

    def regenerate_energy(self, energy: float, fixed: bool = False):
        """恢复能量"""
        if not fixed:
            energy = energy * self.energy_regeneration_rate
        self.energy = min(self.energy + energy, self.max_energy)

    # def add_buff(self, buff: Buff):
    #     """添加buff"""
    #     buff.enable(self)

    def turn_over(self):
        """回合结束"""
        for buff in self.buffs:
            if buff.turn is not None:
                buff.turn -= 1
                if buff.turn <= 0:
                    buff.disable(self)

    def all_attack_action(self, state):
        """获取普攻动作（遍历所有目标配置）"""
        return []

    def all_skill_action(self, state):
        """获取战技动作（遍历所有目标配置）"""
        return []

    def all_ultimate_action(self, state):
        """获取终结技动作（遍历所有目标配置）"""
        return []


class Firefly(CharacterState):
    base_speed = 104
    max_energy = 240

    def all_attack_action(self, state):
        def modify(new):
            new.characters[self.position].distance = FULL_DISTANCE

        a = CharacterAction(state, position=self.position, target=[], reward=1, name=f'{self.position}',
                            modify=modify)
        return [a]


class Bronya(CharacterState):
    base_speed = 99
    max_energy = 120

    def all_skill_action(self, state):
        actions = []
        for target in [0]:
            def modify(new):
                new.characters[self.position].distance = FULL_DISTANCE
                new.characters[target].distance = 0.0

            a = CharacterAction(state, position=self.position, target=[target], reward=0, name=f'{self.position}',
                                modify=modify)
            actions.append(a)
        return actions

    def all_ultimate_action(self, state):
        def modify(new):
            for c in new.characters:
                c.distance = max(0.0, c.distance - 2500.0)
            new.characters[self.position].energy = 0.0

        a = CharacterAction(state, position=self.position, target=[], reward=0, name=f'U{self.position}',
                            modify=modify)
        return [a]


class Timer:
    """倒计时"""

    def __init__(self,
                 speed: float,  # 速度
                 distance: float = FULL_DISTANCE,  # 距离
                 ):
        self.speed = speed  # 速度
        self.distance = distance  # 距离

    @property
    def action_value(self):
        """计算剩余行动值"""
        return self.distance / self.speed

    def actions(self, state):
        """倒计时结束时的动作"""
        return [Action(state)]


class CompleteCombustionTimer(Timer):
    """完全燃烧倒计时"""

    def __init__(self,
                 speed: float = 70,  # 速度
                 distance: float = FULL_DISTANCE,  # 距离
                 ):
        super().__init__(speed, distance)

    def actions(self, state):
        def modify(new: State):
            for c in new.characters:
                if isinstance(c, Firefly):
                    for b in c.buffs:
                        if isinstance(b, CompleteCombustion):
                            b.disable(c)
                            break
                    break
            for t in new.timers:
                if isinstance(t, CompleteCombustionTimer):
                    new.timers.remove(t)
                    break

        return [Action(state, name="CC", modify=modify)]


class State:
    """状态"""
    init_count = 0  # 记录生成的实例数

    def __init__(self,
                 parent=None,  # 父动作
                 sequence: list[CharacterState] = None,  # 角色序列
                 timers: list[Timer] = None,  # 倒计时序列
                 time: float = 0.0,  # 时间
                 in_turn: int | None = None,  # 当前所在回合的角色的编队位置，None为不在回合内
                 ):
        State.init_count += 1
        self.parent: Action | None = parent  # 父动作
        if parent is None:
            self.sequence = sequence  # 角色行动序列
            self.characters = sequence  # 角色编队
            self.timers = timers  # 倒计时序列
            self.time = time  # 时间
            self.in_turn = in_turn  # 当前所在回合的角色的编队位置，None为不在回合内
        else:
            self.sequence = copy.deepcopy(self.parent.parent.sequence)
            self.characters = sorted(self.sequence, key=lambda s: s.position)
            self.timers = copy.deepcopy(self.parent.parent.timers)
            self.time = self.parent.parent.time
            self.in_turn = self.parent.parent.in_turn
            parent.modify(self)
        self.sequence.sort(key=CharacterState.action_value.fget)  # 重排行动序列
        self.timers.sort(key=Timer.action_value.fget)  # 重排倒计时序列
        self._children = None  # 所有子动作
        self._optimal_child = None

    @property
    def children(self):
        """状态的所有可能动作"""
        if self._children is None:
            self._children: list[Action] = self.get_children()
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

    def get_children(self):
        """策略集（获取状态的所有可能动作）"""
        actions: list[Action] = []

        # 若此状态已超时，作为叶子节点返回空列表
        if self.time > MAX_TIME:
            return []
        # 遍历每个位置是否有终结技
        for c in self.characters:
            if c.energy >= c.max_energy:
                actions += c.all_ultimate_action(self)
        # 若在某角色的回合中
        if self.in_turn is not None:
            c = self.characters[self.in_turn]
            # 该角色尚未行动
            if c.distance < 0.01:
                actions += c.all_attack_action(self)
                actions += c.all_skill_action(self)
                return actions
        # 无人行动时推进时间
        a = TimePassAction(self, name='TP')
        actions.append(a)
        return actions


class Action:
    """动作"""
    init_count = 0  # 记录生成的实例数

    def __init__(self,
                 parent: State,
                 name: str = "Unnamed",
                 reward: int = 0,
                 modify=None):
        Action.init_count += 1
        self.parent: State = parent  # 父状态
        self.name = name
        self.reward = reward
        if modify is not None:
            self.modify = modify
        self._child = None  # 子状态
        self._optimal_value = None

    @property
    def child(self):
        """动作后的状态"""
        if self._child is None:
            self._child: State = self.get_child()
        return self._child

    @property
    def optimal_value(self):
        """动作的最优价值"""
        if self._optimal_value is None:
            self._optimal_value: int = self.reward + self.child.optimal_value
        return self._optimal_value

    def __str__(self):
        return f"({self.name}, time: {self.parent.time:.2f})"

    def get_child(self) -> State:
        """状态转移（获取动作后的状态）"""
        return State(self)

    def modify(self, new: State):
        """动作对状态的修改"""
        pass


class TimePassAction(Action):
    """结束当前回合，时间推进至下一个状态"""

    def modify(self, new: State):
        if new.in_turn is not None:
            new.characters[new.in_turn].turn_over()
            new.in_turn = None
            new.sequence.sort(key=CharacterState.action_value.fget)  # 重排行动序列
            new.timers.sort(key=Timer.action_value.fget)  # 重排倒计时序列

        passed_time = new.sequence[0].action_value
        if new.timers:
            passed_time_t = new.timers[0].action_value
            if passed_time > passed_time_t:
                passed_time = passed_time_t
                new.get_children = lambda: new.timers[0].actions(new)
            else:
                new.in_turn = new.sequence[0].position
        else:
            new.in_turn = new.sequence[0].position
        new.time += passed_time
        for c in new.characters:
            c.distance -= passed_time * c.speed
        for t in new.timers:
            t.distance -= passed_time * t.speed


class CharacterAction(Action):
    """角色常规动作（普攻/战技）"""

    def __init__(self,
                 parent: State,
                 position: int,
                 target: list[int] = None,
                 name: str = None,
                 reward: int = 0,
                 modify=None):
        if target is None:
            target = []
        self.position = position
        self.target = target
        super().__init__(parent=parent, reward=reward, name=name, modify=modify)


"""单点情况追踪"""
c0 = Firefly(0, speed=250, distance=FULL_DISTANCE,
             energy=0.0, energy_regeneration_rate=1.0,
             buffs=[CompleteCombustion()])
c1 = Bronya(1, speed=139, distance=FULL_DISTANCE,
            energy=120.0, energy_regeneration_rate=1.0,
            buffs=[])
init_state = State(sequence=[c0, c1], timers=[CompleteCombustionTimer()])
print("State: ", State.init_count)
print("Action: ", Action.init_count)
print(init_state.optimal_value)
print("State: ", State.init_count)
print("Action: ", Action.init_count)

def trace(s, n=0):
    for a in s.children:
        print(' ' * n, a)
        trace(a.child, n + 1)


trace(init_state)

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
start_time = time.time()
round_table = np.zeros([len(sr) for sr in speed_range])

for s0 in range(min_speed[0], max_speed[0] + 1):
    for s1 in range(min_speed[1], max_speed[1] + 1):
        c0 = Firefly(0, speed=s0, distance=FULL_DISTANCE,
                     energy=0.0, energy_regeneration_rate=1.0,
                     buffs=[CompleteCombustion()])
        c1 = Bronya(1, speed=s1, distance=FULL_DISTANCE,
                    energy=120.0, energy_regeneration_rate=1.0,
                    buffs=[])
        init_state = State(sequence=[c0, c1], timers=[CompleteCombustionTimer()])
        t = init_state.optimal_value
        round_table[s0 - min_speed[0]][s1 - min_speed[1]] = t

stop_time = time.time()
print('用时：', stop_time - start_time)

# 保存数据
df = pd.DataFrame(round_table.T, index=speed_range[1], columns=speed_range[0])
csv_filename = "round_table_data.csv"
df.to_csv(csv_filename)
print(f"Round table saved to {csv_filename}")
