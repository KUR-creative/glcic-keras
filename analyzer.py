import sys
from parse import parse

#file_name = './mini_train_log'
#file_name = '1100epoch_128gray_server_log'
#file_name = './19000epoch_128gray_server_log'
#file_name = './mini_240e_log'
file_name = sys.argv[1]

epoch = 0
with open(file_name) as f:
    for li in (line.rstrip('\n') for line in f):
        tmp = parse('epoch {}: [joint loss: {} | mse loss: {}, gan loss: {}]', li)
        if tmp:
            epoch, joint, c_mse, d_bce = tmp

num_epoches = int(epoch) + 1
c_mse_loss = [0] * num_epoches
d_bce_loss = [0] * num_epoches
joint_loss = [0] * num_epoches

#joint_d_loss = []
with open(file_name) as f:
    for li in (line.rstrip('\n') for line in f):
        #print(li)
        tmp = parse('epoch {}: [C mse loss: {}]', li)
        if tmp:
            epoch, c_mse = tmp
            c_mse_loss[int(epoch)] = float(c_mse)
        tmp = parse('epoch {}: [D bce loss: {}]', li)
        if tmp:
            epoch, d_bce = tmp
            d_bce_loss[int(epoch)] = float(d_bce)
        tmp = parse('epoch {}: [joint loss: {} | mse loss: {}, gan loss: {}]', li)
        if tmp:
            epoch, joint, c_mse, d_bce = tmp
            c_mse_loss[int(epoch)] = float(c_mse)
            d_bce_loss[int(epoch)] = float(d_bce)
            joint_loss[int(epoch)] = float(joint)
        #print(tmp)

import matplotlib.pyplot as plt
plt.clf()

plt.subplot(2,1,1)
plt.plot(range(len(c_mse_loss)), c_mse_loss, 'r', label='C mse loss')
plt.plot(range(len(joint_loss)), joint_loss, 'b', label='joint loss')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Error', fontsize=10)
plt.legend(fontsize=10)

plt.subplot(2,1,2)
plt.plot(range(len(d_bce_loss)), d_bce_loss, 'g', label='D bce loss')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Error', fontsize=10)
plt.legend(fontsize=10)

plt.draw()
plt.show()
