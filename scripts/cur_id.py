import os

cur_dir = os.path.abspath(__file__)
for _ in range(2):
    cur_dir = os.path.dirname(cur_dir)

file_name = "cur_id.txt"
file_dir = "exp/sb"
file_dir = os.path.join(cur_dir, file_dir, file_name)

def exp_id_up():
    with open(file_dir,"r") as file:
        cur_val = int(file.read())
    
    cur_val += 1
    with open(file_dir,"w") as file:
        file.write(str(cur_val))
    
    return str(cur_val)

def exp_id_get():
    with open(file_dir,"r") as file:
        cur_val = int(file.read())

    return str(cur_val)

def exp_id_reset():
    with open(file_dir,"w") as file:
        file.write("0")


