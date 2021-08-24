import os 
import time

while True:
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    # print(current_time)
    os.system("git add -A")
    time.sleep(10)
    os.system('git commit -m "Time: {}"'.format(current_time))
    time.sleep(10)
    os.system('git push')
    time.sleep(600)