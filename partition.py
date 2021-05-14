import numpy as np
n=5
big_list = []
for i in range(1, n*n+1):
    big_list.append(i*np.ones(shape = (n,n)))
print(big_list)

def partition(big_list, n):
    return [big_list[i*n:i*n+n] for i in range(n)]

new_list = partition(big_list, n)
print(new_list)
new_list = np.block(new_list)
print(new_list.shape)
print(new_list)