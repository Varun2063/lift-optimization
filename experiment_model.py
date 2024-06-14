import streamlit as st
import logging 
from lift_no_model import move
from model import model_class
from lift_yes_model import move_model

# Class Call to store the stopping variables
class call:
    def __init__(self):
        self.value = False

    def call_stop(self):
        if self.value == True:
            pass
        else:
            self.value = True

class input_streams:
    def __init__(self):
        self.path = ""

    def upload(self, path):
        self.path = path

# class instatiations
model_obj = [model_class() for _ in range(10)]
call_obj = [call() for _ in range(10)]
input_obj = [input_streams() for _ in range(10)]

for i in range(10):
    input_obj[i].upload(f"main_streams\\{i+1}.mp4")

logging.basicConfig(filename='lift_detection.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("lift floors with detection models")
end_floor = []

for i in range(10):
    col1, col2 = st.columns(2)
    state = col1.toggle(f"call on {i}")
    if state:
        logger.info(f"called on {i} floor.")
        end_floor.append(i)
    col2.write(state)

if st.button("run simulation"):
    tim = move_model(call_obj, end_floor)
    st.write(tim)