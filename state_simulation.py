""""""
import copy
import math
import time

import numpy as np
import pandas as pd

# 此版本不再定义Action类，将动作表示为State类中的方法。
# 这使所有State对象构成的树形结构更加典型。
# 在State对象的属性中，包含了Character、Timer两种子对象。
# 在Character对象的属性中，包含了Buff子对象。
# State对象的生成有两种形式：
# 1. 直接生成初始对象。这种方式需要指定对象所有的属性。
# 2. 基于已有对象生成。这种方式会深层复制已有对象（但不复制父子关系），然后再修改部分属性。
# 规定：State对象生成之后（即__init__方法之外），不再修改任何属性。


MAX_TIME = 150.0
HIGH_SPEED_TIME = 142.857
FULL_DISTANCE = 10000.0


class Buff:
    """增益/减益"""
    name = '未命名增益/减益'  # 名称

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


class CompleteCombustion(Buff):
    name = "完全燃烧"

    def enable(self, character):
        character.speed += 60
        character.buffs.append(self)

    def disable(self, character):
        character.speed -= 60
        character.buffs.remove(self)


class Character:
    """角色"""
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

    def turn_over(self):
        """回合结束"""
        # 所有buff持续回合数-1
        for buff in self.buffs:
            if buff.turn is not None:
                buff.turn -= 1
                if buff.turn <= 0:
                    buff.disable(self)

    def attack_actions(self, state):
        """获取普攻动作（遍历所有目标配置）"""
        children: list[State] = []
        return children

    def skill_actions(self, state):
        """获取战技动作（遍历所有目标配置）"""
        children: list[State] = []
        return children

    def ultimate_actions(self, state):
        """获取终结技动作（遍历所有目标配置）"""
        children: list[State] = []
        return children


class Firefly(Character):
    name = '流萤'
    base_speed = 104
    max_energy = 240

    def attack_actions(self, state):
        def modify(new: State):
            new.type = "角色行动后"
            new.characters[self.position].distance = FULL_DISTANCE
            new.value += 1

        return [State(name=f'A{self.position}', parent=state, modify=modify, )]


class Bronya(Character):
    name = '布洛妮娅'
    base_speed = 99
    max_energy = 120

    def skill_actions(self, state):
        children: list[State] = []
        for target in [0]:
            def modify(new: State):
                new.type = "角色行动后"
                new.characters[self.position].distance = FULL_DISTANCE
                new.characters[target].distance = 0.0

            children.append(State(name=f'S{self.position}', parent=state, modify=modify, ))
        return children

    def ultimate_actions(self, state):
        children: list[State] = []
        if self.energy >= self.max_energy:
            def modify(new: State):
                for c in new.sequence:
                    new.type = "角色行动后"
                    c.distance = max(0.0, c.distance - 2500.0)
                new.characters[self.position].energy = 0.0

            children.append(State(name=f'U{self.position}', parent=state, modify=modify, ))
        return children


class Timer:
    """倒计时"""
    name = '未命名倒计时'  # 名称

    def __init__(self,
                 speed: float,  # 速度
                 distance: float = FULL_DISTANCE,  # 距离
                 ):
        self.speed = speed  # 速度
        self.distance = distance  # 距离

    def actions(self, state):
        """倒计时结束时的动作"""
        children: list[State] = []
        return children


class CompleteCombustionTimer(Timer):
    """完全燃烧倒计时"""

    def __init__(self,
                 speed: float = 70,  # 速度
                 distance: float = FULL_DISTANCE,  # 距离
                 ):
        super().__init__(speed, distance)

    def actions(self, state):
        def modify(new: State):
            new.type = "倒计时后"
            for c in new.sequence:
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

        return [State(name=f'CC', parent=state, modify=modify, )]


def action_value(self: Character | Timer) -> float:
    """计算剩余行动值"""
    return self.distance / self.speed


class State:
    """状态"""
    init_count = 0  # 记录生成的实例数

    def __init__(self,
                 name: str,  # 产生此状态的动作名称
                 sequence: list[Character] = None,  # 角色行动序列
                 timers: list[Timer] = None,  # 倒计时序列
                 time: float = 0.0,  # 时间
                 in_turn: int | None = None,  # 当前所在回合的角色的编队位置，None为不在回合内
                 parent=None,  # 父状态
                 modify=None,  # 复制父状态的属性后需要进行的修改（需要修改状态类型、价值）
                 ):
        State.init_count += 1  # 记录生成的实例数
        self.parent: State | None = parent  # 父状态
        self.name: str = name  # 产生此状态的动作名称
        self.type: str = "起点"  # 状态类型（该属性直接由动作决定，不复制父状态，在modify中修改）
        # 所有状态类型：
        # 起点 角色行动前 角色行动后 倒计时前 倒计时后 终结技后
        if parent is None:
            # 指定所有属性
            self.sequence: list[Character] = sorted(sequence, key=action_value)  # 角色行动序列（自动重排）
            self.characters: dict[int, Character] = {c.position: c for c in self.sequence}  # 角色编队 
            self.timers: list[Timer] = sorted(timers, key=action_value)  # 角色行动序列（自动重排）
            self.time: float = time  # 时间
            self.in_turn: int = in_turn  # 当前所在回合的角色的编队位置，None为不在回合内
            self.value = 0  # 价值
        else:
            # 复制父状态的属性
            self.sequence: list[Character] = copy.deepcopy(self.parent.sequence)  # 角色行动序列
            self.characters: dict[int, Character] = {c.position: c for c in self.sequence}  # 角色编队 
            self.timers: list[Timer] = copy.deepcopy(self.parent.timers)  # 倒计时序列
            self.time: float = self.parent.time  # 时间
            self.in_turn: int = self.parent.in_turn  # 当前所在回合的角色的编队位置，None为不在回合内
            self.value = self.parent.value  # 价值（在modify中修改）
            modify(self)
            self.sequence.sort(key=action_value)  # 重排角色行动序列
            self.timers.sort(key=action_value)  # 重排倒计时序列
        self.children = self.get_children()  # 所有子状态

    # @property
    # def children(self):
    #     """所有子状态"""
    #     if self._children is None:
    #         self._children: list[State] = self.get_children()
    #     return self._children

    @property
    def optimal_value(self):
        """子树中的最优价值"""
        if not self.children:
            return self.value
        return max([s.optimal_value for s in self.children])

    def get_children(self):
        """获取所有子状态"""
        children: list[State] = []

        # 若此状态已超时，作为叶子节点返回空列表
        # 由于超时只会发生在时间推进之后，只要时间推进不产生奖励，就可以将叶子节点的价值作为不超时能获得的价值
        if self.time > MAX_TIME:
            return []

        # 若此状态在某倒计时前，则其当前必然在倒计时序列最前
        if self.type == "倒计时前":
            t = self.timers[0]
            # 必须执行倒计时动作（不可插入终结技），返回动作后的状态
            return t.actions(self)

        # 获取每个位置释放（可用的）终结技后的状态
        for c in self.sequence:
            children += c.ultimate_actions(self)

        # 若此状态在某角色行动前，则其当前必然在角色行动序列最前
        if self.type == "角色行动前":
            c = self.sequence[0]
            # 必须选择释放普攻/战技
            children += c.attack_actions(self)
            children += c.skill_actions(self)
            return children

        # 推进时间
        new = self.time_pass_action()
        children.append(new)
        return children

    def time_pass_action(self):
        """退出当前所在回合，时间推进至下一个状态"""

        def modify(new: State):
            # 退出当前所在回合
            if new.in_turn is not None:
                new.characters[new.in_turn].turn_over()
                new.in_turn = None
                new.sequence.sort(key=action_value)  # 重排角色行动序列
                new.timers.sort(key=action_value)  # 重排倒计时序列

            # 比较最靠前的角色和最靠前的倒计时，决定推进的时间和下一个状态的类型
            passed_time = action_value(new.sequence[0])
            new.type = "角色行动前"
            if new.timers:
                passed_time_t = action_value(new.timers[0])
                if passed_time > passed_time_t:
                    passed_time = passed_time_t
                    new.type = "倒计时前"

            # 若为角色行动，则进入该角色的回合
            if new.type == "角色行动前":
                new.in_turn = new.sequence[0].position

            # 推进时间
            new.time += passed_time
            for c in new.sequence:
                c.distance -= passed_time * c.speed
            for t in new.timers:
                t.distance -= passed_time * t.speed

        return State(name="TP", parent=self, modify=modify, )


"""单点情况追踪"""
c0 = Firefly(0, speed=250, distance=FULL_DISTANCE,
             energy=0.0, energy_regeneration_rate=1.0,
             buffs=[CompleteCombustion()])
c1 = Bronya(1, speed=139, distance=FULL_DISTANCE,
            energy=120.0, energy_regeneration_rate=1.0,
            buffs=[])
init_state = State("初始状态", sequence=[c0, c1], timers=[CompleteCombustionTimer()])
print("State: ", State.init_count)
print(init_state.optimal_value)
print("State: ", State.init_count)


def trace(s, n=0):
    for s in s.children:
        print(' ' * n, s.name, f'{s.time:.2f}')
        trace(s, n + 1)


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
        init_state = State("初始状态", sequence=[c0, c1], timers=[CompleteCombustionTimer()])
        t = init_state.optimal_value
        round_table[s0 - min_speed[0]][s1 - min_speed[1]] = t

stop_time = time.time()
print('用时：', stop_time - start_time)

# 保存数据
df = pd.DataFrame(round_table.T, index=speed_range[1], columns=speed_range[0])
csv_filename = "round_table_data.csv"
df.to_csv(csv_filename)
print(f"Round table saved to {csv_filename}")
