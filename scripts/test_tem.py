import time
from unitree_sdk2py.go2.sport.sport_client import SportClient

sport_client = SportClient()
sport_client.Start()

print("SportClient started")

time.sleep(2)

# 让 Go2 起立
sport_client.StandUp()
print("StandUp")

time.sleep(3)

# 停止并蹲下
sport_client.StopMove()
sport_client.StandDown()

sport_client.Stop()
print("Done")
