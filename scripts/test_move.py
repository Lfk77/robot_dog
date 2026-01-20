import time
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# -----------------------------
# 支持的动作命令
# -----------------------------
ACTIONS = {
    "damp": 0,
    "stand_up": 1,
    "stand_down": 2,
    "move_forward": 3,
    "move_lateral": 4,
    "move_rotate": 5,
    "stop_move": 6,
    "hand_stand": 7,
    "balance_stand": 9,
    "recovery": 10,
    "left_flip": 11,
    "back_flip": 12,
    "free_walk": 13,
    "free_bound": 14,
    "free_avoid": 15,
    "walk_upright": 17,
    "cross_step": 18,
    "free_jump": 19
}

def print_help():
    print("Available commands:")
    for k in ACTIONS.keys():
        print("  ", k)
    print("  quit - exit program")

# -----------------------------
# 初始化 DDS 和 Go2 客户端
# -----------------------------
if len(sys.argv) > 1:
    net_if = sys.argv[1]  # 网卡名，例如 enx6c1ff70a9027
    print(f"[INFO] Initializing DDS on interface: {net_if}")
    ChannelFactoryInitialize(0, net_if)
else:
    print("[INFO] Initializing DDS on default interface")
    ChannelFactoryInitialize(0)

sport_client = SportClient()
sport_client.SetTimeout(10.0)
sport_client.Init()
print("[INFO] Go2 SportClient initialized")

# -----------------------------
# 主循环：输入命令控制动作
# -----------------------------
try:
    print_help()
    while True:
        cmd = input("Enter command: ").strip().lower()
        if cmd == "quit":
            print("[INFO] Exiting...")
            break
        if cmd not in ACTIONS:
            print("[WARN] Unknown command")
            print_help()
            continue

        action_id = ACTIONS[cmd]

        if action_id == 0:
            sport_client.Damp()
        elif action_id == 1:
            sport_client.StandUp()
        elif action_id == 2:
            sport_client.StandDown()
        elif action_id == 3:
            sport_client.Move(0.3, 0, 0)  # vx, vy, vyaw
        elif action_id == 4:
            sport_client.Move(0, 0.3, 0)
        elif action_id == 5:
            sport_client.Move(0, 0, 0.5)
        elif action_id == 6:
            sport_client.StopMove()
        elif action_id == 7:
            sport_client.HandStand(True)
            time.sleep(4)
            sport_client.HandStand(False)
        elif action_id == 9:
            sport_client.BalanceStand()
        elif action_id == 10:
            sport_client.RecoveryStand()
        elif action_id == 11:
            sport_client.LeftFlip()
        elif action_id == 12:
            sport_client.BackFlip()
        elif action_id == 13:
            sport_client.FreeWalk()
        elif action_id == 14:
            sport_client.FreeBound(True)
            time.sleep(2)
            sport_client.FreeBound(False)
        elif action_id == 15:
            sport_client.FreeAvoid(True)
            time.sleep(2)
            sport_client.FreeAvoid(False)
        elif action_id == 17:
            sport_client.WalkUpright(True)
            time.sleep(4)
            sport_client.WalkUpright(False)
        elif action_id == 18:
            sport_client.CrossStep(True)
            time.sleep(4)
            sport_client.CrossStep(False)
        elif action_id == 19:
            sport_client.FreeJump(True)
            time.sleep(4)
            sport_client.FreeJump(False)

        print(f"[ACTION] Executed {cmd}")
        time.sleep(1)

finally:
    # 停止运动并安全退出
    sport_client.StopMove()
    sport_client.StandDown()
    print("[INFO] Shutdown complete")
