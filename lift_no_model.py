import time

def move( end_floor ):
    current_floor = 0
    t0 = time.time()
    
    if current_floor < max(end_floor):
        for i in range(current_floor, max(end_floor), 1):
            if i in end_floor:
                time.sleep(6)
            else:
                time.sleep(3)
        current_floor = max(end_floor)
    else:
        for i in range(current_floor, min(end_floor), -1):
            if i in end_floor:
                time.sleep(6)
            else:
                time.sleep(3)
            current_floor = i
        current_floor = min(end_floor)
    t1 = time.time()
    return t1-t0