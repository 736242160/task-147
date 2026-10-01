#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
急诊分诊与容量联动模拟工具（纯 Python 标准库，单文件）。

用法:
    python3 triage.py <输入文件>
    python3 triage.py --demo        # 运行内置示例

输入格式（按行，# 开头为注释）:
    LEVEL    <名称> <优先级> <处置时限分钟>     # 优先级数字越小越紧急
    CAPACITY TOTAL <n>                          # 床位总数
    CAPACITY <级别名> <n>                       # 该级别床位数
    PATIENT  <编号> "<症状描述>" HR=..,RR=..,SPO2=..,SBP=..,T=..
    ARRIVE   <时间分钟> <患者编号>               # 患者到达（触发分诊）
    DISPOSE  <时间分钟> <患者编号>               # 患者被处置（释放床位）
    END      <时间分钟>                          # 模拟结束，做最终超时检查

设计说明
========
分级规则（症状关键词 + 生命体征阈值，取两者中更紧急者）:
    参照 ESI/四级分诊思路：先由生命体征筛查出潜在高危（SpO2、收缩压、
    心率、呼吸率、体温任一越限即升级），再叠加症状关键词（如"昏迷"
    "胸痛"），二者取最高紧急度，避免单一体征正常的高危症状被漏判。
容量策略（升级/降级安置）:
    本级满员时优先"升级安置"——任何患者占用更高级别床位都是安全的
    （高级别床位监护资源更全，向下兼容）；仅当更高级别也满、且患者
    为低危（优先级>=4）时才允许"降级安置"（低危患者对监护需求低，
    占用低资源床位风险可控）；全部满员则进入等待队列并报告错误。
容量配比理由:
    参考急诊流行病学分布，危急重症占比少但需专用抢救床位，中间级别
    流量最大，故按 L1:L2:L3:L4:L5 = 2:4:6:5:3（共 20 床）配置。
"""

import shlex
import sys
from dataclasses import dataclass, field

VITAL_KEYS = ("HR", "RR", "SPO2", "SBP", "T")

# 症状关键词 -> 目标优先级（数字越小越紧急）
KEYWORD_RULES = [
    (1, ["心跳骤停", "呼吸骤停", "昏迷", "休克", "窒息", "严重创伤", "严重烧伤"]),
    (2, ["胸痛", "呼吸困难", "意识模糊", "大出血", "中风", "偏瘫", "抽搐",
         "剧烈头痛", "呕血", "咯血"]),
    (3, ["高热", "剧烈腹痛", "持续呕吐", "骨折", "脱水", "血尿"]),
    (4, ["扭伤", "皮疹", "轻度疼痛", "擦伤", "咽痛"]),
    (5, ["感冒", "咳嗽", "复诊", "开药", "体检"]),
]

# 生命体征阈值规则: (判定函数, 目标优先级)，按紧急度从高到低排列
def _vital_rules(v):
    rules = []
    if v.get("SPO2", 100) < 90 or v.get("SBP", 120) < 90 \
            or v.get("HR", 80) > 130 or v.get("HR", 80) < 40 \
            or v.get("RR", 16) > 30 or v.get("RR", 16) < 8 \
            or v.get("T", 37) >= 41:
        return 1
    if v.get("SPO2", 100) < 94 or v.get("SBP", 120) < 100 \
            or v.get("HR", 80) > 120 or v.get("HR", 80) < 50 \
            or v.get("RR", 16) > 24 or v.get("RR", 16) < 10 \
            or v.get("T", 37) >= 40:
        return 2
    if v.get("SPO2", 100) < 96 or v.get("SBP", 120) < 110 \
            or v.get("HR", 80) > 100 or v.get("RR", 16) > 20 \
            or v.get("T", 37) >= 39:
        return 3
    if v.get("HR", 80) > 90 or v.get("RR", 16) > 16 or v.get("T", 37) >= 37.5:
        return 4
    return 5


@dataclass
class Level:
    name: str
    priority: int
    limit: int          # 处置时限（分钟）
    capacity: int = 0
    occupied: int = 0


@dataclass
class Patient:
    pid: str
    symptoms: str
    vitals: dict


@dataclass
class Stay:
    patient: Patient
    triage_level: Level      # 判定级别
    bed_level: Level = None  # 实际安置床位级别（None=等待中）
    arrive: int = 0
    disposed: int = None
    timeout_reported: bool = False


class TriageSystem:
    def __init__(self):
        self.levels = {}            # 名称 -> Level
        self.patients = {}          # 编号 -> Patient（跨患者状态持久保存）
        self.active = {}            # 编号 -> 当前未处置 Stay
        self.stays = []             # 全部就诊记录（跨患者状态延续）
        self.waiting = []           # 等待床位的 Stay
        self.errors = []
        self.notices = []           # 非错误性事件（如排队后入院）
        self.total_capacity = None
        self.end_time = 0

    # ---------- 分级 ----------
    def triage_priority(self, patient):
        pri = _vital_rules(patient.vitals)
        for target, words in KEYWORD_RULES:
            if any(w in patient.symptoms for w in words):
                pri = min(pri, target)
        return pri

    def level_for_priority(self, pri):
        # 找该优先级的级别；不存在则取最接近的更紧急级别
        cands = sorted(self.levels.values(), key=lambda l: l.priority)
        for lv in cands:
            if lv.priority == pri:
                return lv
        lower = [l for l in cands if l.priority < pri]
        return lower[-1] if lower else cands[0]

    # ---------- 容量联动 ----------
    def place(self, stay):
        """返回 (bed_level, note)；无法安置返回 (None, note)。"""
        own = stay.triage_level
        ordered = sorted(self.levels.values(), key=lambda l: l.priority)
        idx = ordered.index(own)
        # 1) 本级 2) 升级（更紧急级别，资源更全，安全向下兼容）
        candidates = [(own, "本级安置")]
        for lv in reversed(ordered[:idx]):
            candidates.append((lv, "升级安置(本级已满)"))
        # 3) 降级：仅低危患者（优先级>=4）允许
        if own.priority >= 4:
            for lv in ordered[idx + 1:]:
                candidates.append((lv, "降级安置(低危患者,本级及以上已满)"))
        for lv, note in candidates:
            if lv.occupied < lv.capacity:
                lv.occupied += 1
                return lv, note
        return None, "所有可用级别床位已满"

    # ---------- 事件处理 ----------
    def error(self, t, kind, detail):
        self.errors.append("[t=%s] %s: %s" % (t, kind, detail))

    def check_timeouts(self, now):
        for pid, stay in self.active.items():
            limit = stay.triage_level.limit
            if not stay.timeout_reported and now - stay.arrive > limit:
                stay.timeout_reported = True
                self.error(now, "超过处置时限未处置",
                           "患者 %s，级别 %s，时限 %d 分钟，已等待 %d 分钟"
                           % (pid, stay.triage_level.name, limit,
                              now - stay.arrive))

    def arrive(self, t, pid):
        self.check_timeouts(t)
        p = self.patients.get(pid)
        if p is None:
            self.error(t, "患者引用不存在", "ARRIVE 引用了未定义的患者 %s" % pid)
            return
        if pid in self.active:
            self.error(t, "重复分诊",
                       "患者 %s 已在系统中（t=%d 到达，级别 %s），本次到达被忽略"
                       % (pid, self.active[pid].arrive,
                          self.active[pid].triage_level.name))
            return
        pri = self.triage_priority(p)
        lv = self.level_for_priority(pri)
        stay = Stay(patient=p, triage_level=lv, arrive=t)
        bed, note = self.place(stay)
        stay.bed_level = bed
        self.active[pid] = stay
        self.stays.append(stay)
        if bed is None:
            self.waiting.append(stay)
            self.error(t, "容量已满",
                       "患者 %s（级别 %s）无法安置：%s，进入等待队列"
                       % (pid, lv.name, note))

    def dispose(self, t, pid):
        self.check_timeouts(t)
        if pid not in self.patients:
            self.error(t, "患者引用不存在", "DISPOSE 引用了未定义的患者 %s" % pid)
            return
        stay = self.active.pop(pid, None)
        if stay is None:
            self.error(t, "患者引用不存在",
                       "患者 %s 当前无在诊记录，无法处置" % pid)
            return
        stay.disposed = t
        if stay.bed_level is not None:
            stay.bed_level.occupied -= 1
        self.admit_waiting(t)

    def admit_waiting(self, now):
        # 床位释放后按 紧急度优先、同级别先到先得 从等待队列收住
        while self.waiting:
            ordered = sorted(self.waiting,
                             key=lambda s: (s.triage_level.priority, s.arrive))
            for stay in ordered:
                bed, note = self.place(stay)
                if bed is not None:
                    stay.bed_level = bed
                    self.waiting.remove(stay)
                    self.notices.append(
                        "[t=%d] 床位释放，患者 %s 由等待队列收住（%s于 %s）"
                        % (now, stay.patient.pid, note[:2], bed.name))
                    break
            else:
                return

    # ---------- 输入解析 ----------
    def load(self, text):
        events = []
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parts = shlex.split(line)
            except ValueError as e:
                self.error("-", "输入格式错误", "第 %d 行: %s" % (lineno, e))
                continue
            cmd = parts[0].upper()
            try:
                if cmd == "LEVEL":
                    name, pri, limit = parts[1], int(parts[2]), int(parts[3])
                    self.levels[name] = Level(name, pri, limit)
                elif cmd == "CAPACITY":
                    if parts[1].upper() == "TOTAL":
                        self.total_capacity = int(parts[2])
                    else:
                        self.levels[parts[1]].capacity = int(parts[2])
                elif cmd == "PATIENT":
                    pid = parts[1]
                    vitals = {}
                    for kv in parts[-1].split(","):
                        k, _, val = kv.partition("=")
                        vitals[k.strip().upper()] = float(val)
                    symptoms = " ".join(parts[2:-1])
                    self.patients[pid] = Patient(pid, symptoms, vitals)
                elif cmd == "ARRIVE":
                    events.append((int(parts[1]), "ARRIVE", parts[2]))
                elif cmd == "DISPOSE":
                    events.append((int(parts[1]), "DISPOSE", parts[2]))
                elif cmd == "END":
                    events.append((int(parts[1]), "END", None))
                else:
                    self.error("-", "输入格式错误",
                               "第 %d 行: 未知指令 %s" % (lineno, parts[0]))
            except (IndexError, ValueError, KeyError) as e:
                self.error("-", "输入格式错误", "第 %d 行: %s (%s)"
                           % (lineno, raw, e))
        return events

    def run(self, text):
        events = sorted(self.load(text), key=lambda e: e[0])  # 按时间排序，同刻保持文件顺序
        if self.total_capacity is not None:
            s = sum(lv.capacity for lv in self.levels.values())
            if s != self.total_capacity:
                self.error("-", "容量配置不一致",
                           "分级容量之和 %d 与总容量 %d 不符" % (s, self.total_capacity))
        for t, kind, pid in events:
            if kind == "ARRIVE":
                self.arrive(t, pid)
            elif kind == "DISPOSE":
                self.dispose(t, pid)
            elif kind == "END":
                self.end_time = t
                self.check_timeouts(t)
        self.print_report()

    # ---------- 输出 ----------
    def print_report(self):
        print("=" * 72)
        print("分诊结果")
        print("=" * 72)
        header = ("患者", "判定级别", "安置床位", "到达", "处置", "等待分钟", "状态", "症状/体征摘要")
        print("%-6s %-9s %-9s %-5s %-5s %-7s %-8s %s" % header)
        for st in self.stays:
            p = st.patient
            if st.disposed is not None:
                status, disp, wait = "已处置", str(st.disposed), str(st.disposed - st.arrive)
            elif st.bed_level is None:
                status, disp, wait = "等待床位", "-", "-"
            else:
                status, disp, wait = "在诊", "-", "-"
            vit = ",".join("%s=%s" % (k, p.vitals[k]) for k in VITAL_KEYS if k in p.vitals)
            print("%-6s %-9s %-9s %-5s %-5s %-7s %-8s %s [%s]" % (
                p.pid, st.triage_level.name,
                st.bed_level.name if st.bed_level else "无",
                st.arrive, disp, wait, status, p.symptoms, vit))
        if self.notices:
            print()
            print("容量联动事件:")
            for n in self.notices:
                print("  " + n)
        print()
        print("=" * 72)
        print("错误报告（共 %d 条）" % len(self.errors))
        print("=" * 72)
        if not self.errors:
            print("无错误。")
        for i, e in enumerate(self.errors, 1):
            print("%2d. %s" % (i, e))
        print()
        print("床位占用（结束时）: " + ", ".join(
            "%s %d/%d" % (lv.name, lv.occupied, lv.capacity)
            for lv in sorted(self.levels.values(), key=lambda l: l.priority)))


DEMO = r"""
# 级别定义: 名称 优先级 处置时限(分钟)，优先级数字越小越紧急
LEVEL 一级-危急 1 10
LEVEL 二级-危重 2 30
LEVEL 三级-急症 3 60
LEVEL 四级-次急 4 120
LEVEL 五级-非急 5 240
# 容量: 总 8 床，按 1:1:2:2:2 配比（示例规模；生产可按 2:4:6:5:3/20 床放大）
CAPACITY TOTAL 8
CAPACITY 一级-危急 1
CAPACITY 二级-危重 1
CAPACITY 三级-急症 2
CAPACITY 四级-次急 2
CAPACITY 五级-非急 2
# 患者: 编号 "症状描述" 生命体征 HR RR SPO2 SBP T
PATIENT P001 "胸痛 大汗 呼吸困难" HR=128,RR=26,SPO2=93,SBP=95,T=37.2
PATIENT P002 "感冒 咳嗽" HR=82,RR=16,SPO2=98,SBP=120,T=37.0
PATIENT P003 "昏迷 休克" HR=135,RR=32,SPO2=85,SBP=70,T=36.0
PATIENT P004 "扭伤 轻度疼痛" HR=88,RR=17,SPO2=99,SBP=125,T=36.8
PATIENT P005 "高热 持续呕吐" HR=108,RR=22,SPO2=97,SBP=115,T=39.5
PATIENT P006 "心跳骤停" HR=0,RR=0,SPO2=60,SBP=50,T=35.0
PATIENT P007 "皮疹 瘙痒" HR=76,RR=15,SPO2=99,SBP=118,T=36.6
PATIENT P008 "复诊 开药" HR=72,RR=14,SPO2=99,SBP=122,T=36.5
PATIENT P010 "剧烈腹痛" HR=86,RR=18,SPO2=98,SBP=120,T=37.1
PATIENT P011 "擦伤" HR=80,RR=16,SPO2=99,SBP=120,T=36.7
# 事件流（时间单位: 分钟）
ARRIVE 0 P001
ARRIVE 2 P003
ARRIVE 5 P006
ARRIVE 6 P002
ARRIVE 8 P004
ARRIVE 10 P005
ARRIVE 12 P007
ARRIVE 15 P008
ARRIVE 18 P010
ARRIVE 22 P011
ARRIVE 20 P001
ARRIVE 30 P999
DISPOSE 25 P003
DISPOSE 40 P006
DISPOSE 45 P001
DISPOSE 70 P009
DISPOSE 200 P004
END 300
"""


def main(argv):
    if len(argv) == 2 and argv[1] == "--demo":
        text = DEMO
    elif len(argv) == 2:
        with open(argv[1], encoding="utf-8") as f:
            text = f.read()
    else:
        print(__doc__)
        return 1
    TriageSystem().run(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
