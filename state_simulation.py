""""""
import copy
import math
import time

import numpy as np
import pandas as pd
from typing_extensions import Callable

"""
每个状态State包含以下属性：
（战斗相关）
角色编队 characters : dict[int, Character]
倒计时表 timers : dict[int, Timer]
当前回合 in_turn : int
行动序列 sequence : list[int]
战技点 skill_point : int
时间（行动值） time : float
类型 type : str
（搜索相关）
父节点 parent : State
子节点 children : list[State]
名称 name : str
累积价值 value : float
最优价值 optimal_value : float
注：行动序列不含当前回合的角色/倒计时。
注：名称记录具体节点信息。
注：最优价值为子树中最大的累积价值。

状态转移的过程称为行动Action，根据状态前的行动，状态分类为以下几种：
起点
角色行动后（普攻/战技）
倒计时后
敌人行动后（TODO）
终结技后
追加行动后（TODO）
时间推进后（回合切换）
注：每个状态一定处于某一角色/倒计时/敌人的回合中，故回合结束与回合开始可以合并为一种行动。
注：终结技可以插入在起点/角色回合开始/角色行动/倒计时/敌人行动/终结技之后，但需要等待这些行动触发的追加行动结束。

每个角色Character包含以下属性：
（基础属性）
名称 name : str
基础速度 base_speed : int
能量上限 max_energy : int
能量消耗 energy_cost : int
位置 position : int
...
（可变属性）
速度 speed : float
距离 distance : float
能量 energy: float
能量恢复效率 energy_regeneration_rate: float
...
buff表 buffs : list[Buff]

每个Buff包含以下属性：
剩余回合数 turn : int
层数 stack : int

每个倒计时Timer包含以下属性：
位置 position : int
速度 speed : float
距离 distance : float
"""

MAX_TIME = 150.0
FULL_DISTANCE = 10000.0


class Buff:
    """增益/减益"""
    name: str = '未命名增益/减益'  # 名称
    decrease_at_start: bool = False  # 剩余回合数是否在回合开始减少

    def __init__(self,
                 turn: int | None = None,  # 剩余回合数（None为永续）
                 stack: int | None = None,  # 层数（None为不可叠层）
                 ):
        self.turn: int | None = turn  # 剩余回合数
        self.stack: int | None = stack  # 层数

    def enable(self, character: 'Character'):
        """buff效果"""
        pass
        # 标准模板（无效果）

    def disable(self, character: 'Character'):
        """删除buff效果"""
        pass
        # 标准模板（无效果）


class CompleteCombustionBuff(Buff):
    name = "完全燃烧"

    def enable(self, character: 'Character'):
        character.speed += 60

    def disable(self, character: 'Character'):
        character.speed -= 60


class SylvanEnigma2Buff(Buff):
    name = "涂山玄设2"  # 战技点恢复效果


class SomatotypicalHelixBuff(Buff):
    name = "分型的螺旋"

    def enable(self, character: 'Character'):
        character.speed += 0.098 * character.base_speed

    def disable(self, character: 'Character'):
        character.speed -= 0.098 * character.base_speed


class TorridScorchBuff(Buff):
    name = "炽灼"
    decrease_at_start = True

    def __init__(self):
        super().__init__(3, None)


class OvertoneBuff(Buff):
    name = "弦外音"
    decrease_at_start = True

    def __init__(self):
        super().__init__(3, None)


class Character:
    """角色"""
    # 以下是战斗中不变的属性
    name: str = '未命名角色'  # 名称
    # combat_type: int = 1  # 战斗属性（1-7：物火冰雷风量虚）
    # path: int = 1  # 命途（1-8：毁灭 巡猎 智识 同协 虚无 存护 丰饶 记忆）
    # level: int = 80  # 等级
    # base_max_hp: int = 2000  # 基础生命上限
    # base_attack: int = 1000  # 基础攻击力
    # base_defence: int = 1000  # 基础防御力
    base_speed: int = 100  # 基础速度
    max_energy: int = 100  # 能量上限
    position: int = 0  # 位置

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
        self.__class__.position = position  # 编队位置
        # self.max_hp: float = max_hp  # 生命上限
        # self.hp: float = hp  # 生命值
        # self.attack: float = attack  # 攻击力
        # self.defence: float = defence  # 防御力
        self.speed: float = speed  # 速度
        self.distance: float = distance  # 距离
        self.energy: float = energy  # 能量
        self.energy_regeneration_rate: float = energy_regeneration_rate  # 能量恢复效率
        # self.crit_rate: float = crit_rate  # 暴击率
        # self.crit_damage: float = crit_damage  # 暴击伤害
        # self.break_effect: float = break_effect  # 击破特攻
        # self.outgoing_healing_boost: float = outgoing_healing_boost  # 治疗量加成
        # self.effect_hit_rate: float = effect_hit_rate  # 效果命中
        # self.effect_resistance: float = effect_resistance  # 效果抵抗
        # self.damage_boost = damage_boost: list[float]  # 伤害提高
        # self.resistance_boost = resistance_boost: list[float]  # 抗性提高
        self.buffs: list[Buff] = buffs  # 增益/减益效果

    def regenerate_energy(self, energy: float, fixed: bool = False):
        """恢复能量"""
        if not fixed:
            energy = energy * self.energy_regeneration_rate
        self.energy = min(self.energy + energy, self.max_energy)

    def get_buff(self, buff_class: type[Buff]) -> Buff | None:
        """寻找某一类型的buff（只返回第一个）"""
        for b in self.buffs:
            if isinstance(b, buff_class):
                return b
        else:
            return None

    def append_buff(self, buff_class: type[Buff]):
        """添加某一类型的buff"""
        b = buff_class()
        b.enable(self)
        self.buffs.append(b)

    def remove_buff(self, buff_class: type[Buff]):
        """删除某一类型的buff（只删除第一个），没有时不删除"""
        b = self.get_buff(buff_class)
        if b is not None:
            b.disable(self)
            self.buffs.remove(b)

    def turn_start(self):
        """回合开始"""
        # 所有buff持续回合数-1
        for b in self.buffs:
            if b.turn is not None and b.decrease_at_start:  # 非永续且在回合开始结算
                b.turn -= 1
                if b.turn <= 0:
                    b.disable(self)

    def turn_over(self):
        """回合结束"""
        # 所有buff持续回合数-1
        for b in self.buffs:
            if b.turn is not None and not b.decrease_at_start:  # 非永续且在回合结束结算
                b.turn -= 1
                if b.turn <= 0:
                    b.disable(self)

    def attack_modify(self, new: 'State'):
        """普攻动作对状态的修改"""
        new.skill_point = min(new.skill_point + 1, new.max_skill_point)
        new_self = new.characters[self.position]
        new_self.distance = FULL_DISTANCE
        new_self.regenerate_energy(20.0)
        # 标准模板

    def attack_actions(self, state: 'State') -> list['State']:
        """获取普攻动作（遍历所有目标配置）"""
        children: list[State] = []
        children.append(State(name=f'{self.position}A', type="角色行动后",
                              parent=state, modify=self.attack_modify, ))
        return children
        # 标准模板

    def skill_modify(self, new: 'State'):
        """战技动作对状态的修改"""
        new.skill_point -= 1
        new_self = new.characters[self.position]
        new_self.distance = FULL_DISTANCE
        new_self.regenerate_energy(30.0)
        # 标准模板

    def skill_actions(self, state: 'State') -> list['State']:
        """获取战技动作（遍历所有目标配置）"""
        children: list[State] = []
        if state.skill_point >= 1:
            children.append(State(name=f'{self.position}S', type="角色行动后",
                                  parent=state, modify=self.skill_modify, ))
        return children
        # 标准模板

    def ultimate_modify(self, new: 'State'):
        """终结技动作对状态的修改"""
        new_self = new.characters[self.position]
        new_self.energy = 0.0
        new_self.regenerate_energy(5.0)
        # 标准模板

    def ultimate_actions(self, state: 'State') -> list['State']:
        """获取终结技动作（遍历所有目标配置）"""
        children: list[State] = []
        if self.energy >= self.max_energy:
            children.append(State(name=f'{self.position}U', type="终结技后",
                                  parent=state, modify=self.ultimate_modify, ))
        return children
        # 标准模板


class Firefly(Character):
    name = '流萤'
    base_speed = 104
    max_energy = 240

    def enhanced_attack_modify(self, new: 'State'):
        new.skill_point = min(new.skill_point + 1, new.max_skill_point)
        new_self = new.characters[self.position]
        new_self.distance = FULL_DISTANCE

    def attack_actions(self, state: 'State') -> list['State']:
        children: list[State] = []
        if self.get_buff(CompleteCombustionBuff) is None:
            children.append(State(name=f'{self.position}A', type="角色行动后",
                                  parent=state, modify=self.attack_modify, ))
        else:
            children.append(State(name=f'{self.position}A+', type="角色行动后",
                                  parent=state, modify=self.enhanced_attack_modify, ))
        return children

    def skill_modify(self, new: 'State'):
        new.skill_point -= 1
        new_self = new.characters[self.position]
        new_self.distance = FULL_DISTANCE * (1.0 - 0.25)
        new_self.regenerate_energy(0.6 * self.max_energy, fixed=True)

    def enhanced_skill_modify(self, new: 'State'):
        """强化战技动作对状态的修改"""
        new.skill_point -= 1
        new_self = new.characters[self.position]
        new_self.distance = FULL_DISTANCE
        new.value += 1

    def skill_actions(self, state: 'State') -> list['State']:
        children: list[State] = []
        if state.skill_point >= 1:
            if self.get_buff(CompleteCombustionBuff) is None:
                children.append(State(name=f'{self.position}S', type="角色行动后",
                                      parent=state, modify=self.skill_modify, ))
            else:
                children.append(State(name=f'{self.position}S+', type="角色行动后",
                                      parent=state, modify=self.enhanced_skill_modify, ))
        return children

    def ultimate_modify(self, new: 'State'):
        new_self = new.characters[self.position]
        new_self.energy = 0.0
        new_self.regenerate_energy(5.0)
        new_self.append_buff(CompleteCombustionBuff)
        new_self.distance = max(0.0, new_self.distance - FULL_DISTANCE)
        i = max(10, max(new.characters_and_timers.keys()))
        new.timers[i + 1] = CompleteCombustionTimer(i + 1)

    def ultimate_actions(self, state: 'State') -> list['State']:
        children: list[State] = []
        if self.get_buff(CompleteCombustionBuff) is None and self.energy >= self.max_energy:
            children.append(State(name=f'{self.position}U', type="终结技后",
                                  parent=state, modify=self.ultimate_modify, ))
        return children


class Bronya(Character):
    name = '布洛妮娅'
    base_speed = 99
    max_energy = 120

    def attack_modify(self, new: 'State'):
        new.skill_point = min(new.skill_point + 1, new.max_skill_point)
        new_self = new.characters[self.position]
        new_self.distance = FULL_DISTANCE * (1.0 - 0.28)
        new_self.regenerate_energy(20.0)

    def skill_actions(self, state: 'State') -> list['State']:
        children: list[State] = []
        if state.skill_point >= 1:
            # for target_position in state.characters:
            for target_position in [1]:  # 剪枝：只拉1号位
                if state.characters[target_position].distance < 100.0:  # 剪枝：拒绝短距离拉条
                    continue

                def modify(new: State):
                    new.skill_point -= 1
                    new_self = new.characters[self.position]
                    new_self.distance = FULL_DISTANCE
                    new_self.regenerate_energy(30.0)
                    if target_position != self.position:
                        new_target = new.characters[target_position]
                        new_target.distance = 0.0  # 立即行动
                        # 将目标角色提前到行动序列最前方
                        new.sequence.remove(target_position)
                        new.sequence.insert(0, target_position)

                children.append(State(name=f'{self.position}S{target_position}', type="角色行动后",
                                      parent=state, modify=modify, ))
        return children

    def ultimate_modify(self, new: 'State'):
        new_self = new.characters[self.position]
        new_self.energy = 0.0
        new_self.regenerate_energy(5.0)
        for c in new.characters.values():
            c.distance = max(0.0, c.distance - 0.22 * FULL_DISTANCE)


class Fugue(Character):
    name = '忘归人'
    base_speed = 102
    max_energy = 130

    def attack_actions(self, state: 'State') -> list['State']:
        children: list[State] = []
        if self.get_buff(TorridScorchBuff) is not None:  # 剪枝：只在有战技buff时使用普攻（不允许断buff）
            children.append(State(name=f'{self.position}A', type="角色行动后",
                                  parent=state, modify=self.attack_modify, ))
        return children

    def skill_modify(self, new: 'State'):
        new.skill_point -= 1
        new_self = new.characters[self.position]
        new_self.distance = FULL_DISTANCE
        new_self.regenerate_energy(30.0)
        b = new_self.get_buff(SylvanEnigma2Buff)
        if b is not None:
            new.skill_point += 1
            b.disable(new_self)
        new_self.remove_buff(TorridScorchBuff)
        new_self.append_buff(TorridScorchBuff)

    def skill_actions(self, state: 'State') -> list['State']:
        children: list[State] = []
        if self.get_buff(TorridScorchBuff) is None:  # 剪枝：只在战技buff结束后使用战技
            if state.skill_point >= 1:
                children.append(State(name=f'{self.position}S', type="角色行动后",
                                      parent=state, modify=self.skill_modify, ))
        return children

    def ultimate_actions(self, state: 'State') -> list['State']:
        return []  # 剪枝：忽略终结技


class RuanMei(Character):
    name = '阮梅'
    base_speed = 104
    max_energy = 130

    def attack_actions(self, state: 'State') -> list['State']:
        children: list[State] = []
        if self.get_buff(OvertoneBuff) is not None:  # 剪枝：只在有战技buff时使用普攻（不允许断buff）
            children.append(State(name=f'{self.position}A', type="角色行动后",
                                  parent=state, modify=self.attack_modify, ))
        return children

    def skill_modify(self, new: 'State'):
        new.skill_point -= 1
        new_self = new.characters[self.position]
        new_self.distance = FULL_DISTANCE
        new_self.regenerate_energy(30.0)
        new_self.remove_buff(OvertoneBuff)
        new_self.append_buff(OvertoneBuff)

    def skill_actions(self, state: 'State') -> list['State']:
        children: list[State] = []
        if self.get_buff(OvertoneBuff) is None:  # 剪枝：只在战技buff结束后使用战技
            if state.skill_point >= 1:
                children.append(State(name=f'{self.position}S', type="角色行动后",
                                      parent=state, modify=self.skill_modify, ))
        return children

    def ultimate_actions(self, state: 'State') -> list['State']:
        return []  # 剪枝：忽略终结技


class Timer:
    """倒计时"""
    name: str = '未命名倒计时'  # 名称

    def __init__(self,
                 position: int,  # 位置
                 speed: float,  # 速度
                 distance: float = FULL_DISTANCE,  # 距离
                 ):
        self.position: int = position  # 位置
        self.speed: float = speed  # 速度
        self.distance: float = distance  # 距离

    def modify(self, new: 'State'):
        """倒计时结束时对状态的修改"""
        pass
        # 标准模板

    def actions(self, state: 'State') -> list['State']:
        """倒计时结束时的动作"""
        children: list[State] = []
        children.append(State(name=f'Timer', type="倒计时后", parent=state, modify=self.modify, ))
        return children
        # 标准模板

    def turn_over(self, state: 'State'):
        """在序列中删除倒计时（倒计时回合结束后执行）"""
        state.timers.pop(self.position)
        state.sequence.remove(self.position)


class CompleteCombustionTimer(Timer):
    """完全燃烧倒计时"""

    def __init__(self,
                 position: int,  # 位置
                 speed: float = 70,  # 速度
                 distance: float = FULL_DISTANCE,  # 距离
                 ):
        super().__init__(position, speed, distance)

    def modify(self, new: 'State'):
        for c in new.characters.values():
            if isinstance(c, Firefly):
                c.remove_buff(CompleteCombustionBuff)
                break

    def actions(self, state: 'State') -> list['State']:
        return [State(name=f'CC', type="倒计时后", parent=state, modify=self.modify, )]


def action_value(self: Character | Timer) -> float:
    """计算剩余行动值"""
    return self.distance / self.speed


def time_pass_modify(new: 'State'):
    # 退出当前所在回合
    if new.in_turn is not None:
        i = new.in_turn
        new.sequence.append(i)
        new.in_turn = None
        if i in new.characters.keys():
            new.characters[i].turn_over()
        if i in new.timers.keys():
            new.timers[i].turn_over(new)
        new.sort_sequence()  # 重排行动序列

    # 推进时间
    passed_time = action_value(new.characters_and_timers[new.sequence[0]])
    new.time += passed_time
    for c in new.characters.values():
        c.distance -= passed_time * c.speed
    for t in new.timers.values():
        t.distance -= passed_time * t.speed

    # 进入新回合
    new.in_turn = new.sequence.pop(0)
    if new.in_turn in new.characters.keys():
        new.characters[new.in_turn].turn_start()


class State:
    """状态"""
    init_count: int = 0  # 记录生成的实例数
    max_skill_point: int = 5  # 最大战技点

    def __init__(self,
                 name: str,  # 产生此状态的动作名称
                 type: str,  # 状态类型
                 characters: dict[int, Character] = None,  # 角色编队
                 timers: dict[int, Timer] = None,  # 倒计时表
                 in_turn: int | None = None,  # 当前回合的角色/倒计时的编号，None为不在回合内（仅允许起点不在回合内）
                 time: float = 0.0,  # 时间
                 skill_point: int = 3,  # 战技点
                 parent: 'State | None' = None,  # 父节点
                 modify: Callable[['State'], None] = None,  # 复制父节点的属性后需要进行的修改（需要修改状态类型、价值）
                 ):
        self.__class__.init_count += 1  # 记录生成的实例数
        self.parent: State | None = parent  # 父节点
        self.name: str = name  # 产生此状态的动作名称
        self.type: str = type  # 状态类型（该属性直接由动作决定）
        # 所有状态类型：
        # 起点 角色行动后 倒计时后 终结技后 时间推进后
        if parent is None:
            # 指定所有属性
            self.characters: dict[int, Character] = characters  # 角色编队
            self.timers: dict[int, Timer] = timers  # 倒计时表
            self.characters_and_timers = self.characters | self.timers
            self.in_turn: int = in_turn  # 当前所在回合的角色的编队位置，None为不在回合内（仅允许起点不在回合内）
            self.sequence: list[int] = list(characters.keys() | timers.keys())  # 行动序列
            self.sort_sequence()  # 重排行动序列
            self.time: float = time  # 时间
            self.skill_point: int = skill_point  # 战技点
            self.value = 0  # 价值
        else:
            # 复制父节点的属性
            self.characters: dict[int, Character] = copy.deepcopy(self.parent.characters)  # 角色编队
            self.timers: dict[int, Timer] = copy.deepcopy(self.parent.timers)  # 倒计时表
            self.characters_and_timers = self.characters | self.timers
            self.in_turn: int = self.parent.in_turn  # 当前所在回合的角色的编队位置，None为不在回合内（仅允许起点不在回合内）
            self.sequence: list[int] = copy.deepcopy(self.parent.sequence)  # 行动序列
            self.time: float = self.parent.time  # 时间
            self.skill_point: int = self.parent.skill_point  # 战技点
            self.value = self.parent.value  # 价值（在modify中修改）
            modify(self)
            self.sort_sequence()  # 重排行动序列
        print(self.value, end='')
        self.children: list[State] = self.get_children()  # 所有子节点

    @property
    def optimal_value(self):
        """子树中的最优价值"""
        if not self.children:
            return self.value
        return max([s.optimal_value for s in self.children])

    def sort_sequence(self):
        """重排行动序列"""
        self.sequence.sort(key=lambda x: action_value(self.characters_and_timers[x]))

    def get_children(self) -> list['State']:
        """获取所有子节点"""
        children: list[State] = []

        # 若此状态已超时，则返回空列表
        if self.time > MAX_TIME:
            return []

        # 若此状态在时间推进后（回合开始），且处于倒计时的回合
        if self.type == "时间推进后" and self.in_turn in self.timers.keys():
            # 必须执行倒计时动作（不可插入终结技）
            t = self.timers[self.in_turn]
            return t.actions(self)

        # 获取每个位置释放（可用的）终结技后的状态
        for c in self.characters.values():
            children += c.ultimate_actions(self)

        # 若此状态在时间推进后（回合开始），且处于角色的回合
        if self.type == "时间推进后" and self.in_turn in self.characters.keys():
            # 必须选择释放普攻/战技
            c = self.characters[self.in_turn]
            children += c.attack_actions(self)
            children += c.skill_actions(self)
            return children

        # 推进时间
        children += self.time_pass_actions()
        return children

    def time_pass_actions(self) -> list['State']:
        """退出当前所在回合，时间推进至下一个状态"""
        new = State(name="TP", type="时间推进后", parent=self, modify=time_pass_modify, )

        # 若此状态已超时，则返回空列表
        if new.time > MAX_TIME:
            return []

        return [new]


"""单点情况追踪"""

start_time = time.time()
c1 = Firefly(1, speed=250, distance=FULL_DISTANCE,
             energy=0.0, energy_regeneration_rate=1.0,
             buffs=[CompleteCombustionBuff()])
c2 = Bronya(2, speed=139, distance=FULL_DISTANCE,
            energy=120.0, energy_regeneration_rate=1.244,
            buffs=[])
c3 = Fugue(3, speed=194.0, distance=FULL_DISTANCE,
           energy=65.0, energy_regeneration_rate=1.0,
           buffs=[SylvanEnigma2Buff()])
c4 = RuanMei(4, speed=140.0, distance=FULL_DISTANCE,
             energy=65.0, energy_regeneration_rate=1.244,
             buffs=[])
characters = {c.position: c for c in [c1, c2, c3, c4]}
for c in characters.values():
    c.append_buff(SomatotypicalHelixBuff)
c4.append_buff(OvertoneBuff)
init_state = State("初始状态", type="起点",
                   characters=characters, timers={11: CompleteCombustionTimer(11)})
print(init_state.optimal_value)
print("State: ", State.init_count)


def trace(s, n=0, log=''):
    if not s.children:
        print(log)
    for ss in reversed(s.children):
        if ss.optimal_value == s.optimal_value:
            # print(log, ss.name, ' ', f'{s.time:.2f}', f'{s.skill_point}', f'{s.value}')
            if ss.name == "TP":
                trace(ss, n + 1, log)
            else:
                trace(ss, n + 1, log + ' ' + ss.name)


trace(init_state)

stop_time = time.time()
print('用时：', stop_time - start_time)

"""扫描参数空间"""
# # 流萤基础速度：104
# # 布洛妮娅基础速度：99
# basic_speed = [104, 99]
# # 固定加速（行迹属性、非遗器效果）
# fixed_speed_add = [5 + 60, 0]
# fixed_speed_boost = [0.098, 0.098]  # 阮梅加速
# # 遗器加速理论范围
# min_speed_add = 0.0
# min_speed_boost = -0.08
# max_speed_add = 25 + 2.6 * 30
# max_speed_boost = 0.06 * 3
# # 计算速度范围
# min_speed = [math.ceil(bs * (1 + fsb + min_speed_boost) + fsa + min_speed_add)
#              for bs, fsa, fsb in zip(basic_speed, fixed_speed_add, fixed_speed_boost)]
# max_speed = [math.floor(bs * (1 + fsb + max_speed_boost) + fsa + max_speed_add)
#              for bs, fsa, fsb in zip(basic_speed, fixed_speed_add, fixed_speed_boost)]
# speed_range = [np.arange(a, b + 1) for a, b in zip(min_speed, max_speed)]
#
# # 遍历所有速度配置
# start_time = time.time()
# round_count = 0
# round_table = np.zeros([len(sr) for sr in speed_range])
# max_round_count = round_table.size
#
# for s0 in range(min_speed[0], max_speed[0] + 1):
#     for s1 in range(min_speed[1], max_speed[1] + 1):
#         c0 = Firefly(0, speed=s0, distance=FULL_DISTANCE,
#                      energy=0.0, energy_regeneration_rate=1.0,
#                      buffs=[CompleteCombustionBuff()])
#         c1 = Bronya(1, speed=s1, distance=FULL_DISTANCE,
#                     energy=120.0, energy_regeneration_rate=1.0,
#                     buffs=[])
#         init_state = State("初始状态", sequence=[c0, c1], timers=[CompleteCombustionTimer()], skill_point=5)
#         t = init_state.optimal_value
#         round_table[s0 - min_speed[0]][s1 - min_speed[1]] = t
#         round_count += 1
#         if round_count % 10 == 0:
#             print(f"已完成：{round_count}/{max_round_count}")
#
# stop_time = time.time()
# print('用时：', stop_time - start_time)
#
# # 保存数据
# df = pd.DataFrame(round_table.T, index=speed_range[1], columns=speed_range[0])
# csv_filename = "round_table_data.csv"
# df.to_csv(csv_filename)
# print(f"Round table saved to {csv_filename}")
