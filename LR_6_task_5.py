import numpy as np
import neurolab as nl

# Букви G, V, F
target = [
    [0, -1, -1, -1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0],  # G
    [-1, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0],  # V
    [-1, -1, -1, 0, 0, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0],  # F
]

chars = ['G', 'V', 'F']
target = np.asfarray(target)
target[target == 0] = -1

net = nl.net.newhop(target)
output = net.sim(target)
print("Тест на навчальних зразках:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

print("\nТест на зіпсованій букві G:")
test = np.asfarray([0, -1, -1, -1, 0, -1, 1, 0, 0, 0, -1, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, -1, -1, 0]) 
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[0]).all(), 'Кількість кроків симуляції', len(net.layers[0].outs))
