import pickle
import torch
import numpy as np

def my_fn(x):
    print(x)

def apply_along_axis(function, x, axis: int = 0):
    return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=axis)], dim=axis)

def find_rows(source, target):
    return np.where((source == target).all(axis=1))[0]

def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


with open('/home/chanho/l_features.pkl', 'rb') as f:
    l_features = pickle.load(f)

with open('/home/chanho/l_xyz.pkl', 'rb') as f:
    l_xyz = pickle.load(f)





# print(l_xyz[1].shape)
# print(l_xyz[3])

# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# b = np.array([[1,2,3],[7,8,9]])
# print(np.all(np.isin(a,b), axis=1))

looking = np.array([[10.1, 20.1, 30.1], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1]])

Y = np.array([[1.1, 2.1, 3.1],
              [10.1, 20.1, 30.1],
              [100.1, 200.1, 300.1],
              ])




import time
a = l_xyz[1][1].cpu().numpy()
b = l_xyz[3][1].cpu().numpy()

c = np.apply_along_axis(lambda x: np.where((a == x).all(axis=1)), axis=1, arr=b).squeeze()

a = l_xyz[1]
b = l_xyz[3]


n = time.time()
for i in range(4):
    c = apply_along_axis(lambda x: torch.where((a[i]==x).all(axis=1) == True)[0], b[i]).squeeze()
print(c)
print(time.time() - n)




# c = apply_along_axis(lambda x: torch.where((a==x).all(axis=2) == True)[0], b, axis=1).squeeze()
# print(c)




# a = l_xyz[1][2]
# b = l_xyz[3][2]




# for i in range(4):
#     print(torch.where())



# np.where(np.all(np.isin(l_xyz[2][0].cpu().numpy(), l_xyz[3][0].cpu().numpy()), axis=1) == True)[0]


# print(l_xyz[1])
# print(l_xyz[1][:,1])
# print(torch.where(l_xyz[1][:, 1] == l_xyz[3][:, 1]))







        # find_4096idx_from_512idx = []

        # for batch in range(4):
        #     find_4096idx_from_512idx.append(sample1024[batch][sample512[batch].tolist()].tolist())


        # print(feature4096[0][:, find_4096idx_from_512idx[0]].shape)


        # # find_4096idx_from_512idx[batch][idx] idx: 0~511
        # # xyz4096[batch][idx] idx: 0~4095
        # # xyz512[batch][idx] idx: 0~511

        # print(xyz4096[0][find_4096idx_from_512idx[0][0]])     
        # print(xyz512[0][0])


        # print(xyz4096[1][find_4096idx_from_512idx[1][0]])
        # print(xyz512[1][0])
        
        # print(xyz4096[2][find_4096idx_from_512idx[2][0]])
        # print(xyz512[2][0])

        # print(xyz4096[3][find_4096idx_from_512idx[3][0]])
        # print(xyz512[3][0])

        # for batch in range(4):
        #     sample1024[batch][sample512[batch].tolist()]

        
        
        # # print(l_xyz[1][0][sample1024[0][sample512[0][0]]])
        # # print(l_xyz[3][0][0])

