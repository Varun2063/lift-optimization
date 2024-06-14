import time
from model import model_class



def move_model(call_obj, end_floor):

    class input_streams:
        def __init__(self):
            self.path = ""
    
        def upload(self, path):
            self.path = path
    
    # class instatiations
    input_obj = [input_streams() for _ in range(10)]
    model_obj = [model_class() for _ in range(10)]
    
    for i in range(10):
        input_obj[i].upload(f"main_streams\\{i+1}.mp4")
    
    current_floor = 0
    t0 = time.time()
    
    for i in end_floor:
        call_obj[i].value = model_obj[i].model_yolo(input_obj[i].path)

    if current_floor < max(end_floor):
        for i in range(current_floor, max(end_floor), 1):
            if ( i in end_floor or call_obj[i].value == True):
                time.sleep(6)
            else:
                time.sleep(3)
        current_floor = max(end_floor)
    else:
        for i in range(current_floor, min(end_floor), -1):
            if (i in end_floor or call_obj[i].value == True):
                time.sleep(6)
            else:
                time.sleep(3)
            current_floor = i
        current_floor = min(end_floor)
    t1 = time.time()
    return t1-t0