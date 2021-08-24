import os 
import time

while True:
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    # print(current_time)
    os.system("git add m='Time:{}'".format(current_time))
    time.sleep(600)