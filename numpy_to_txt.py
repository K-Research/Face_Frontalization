import numpy as np

X_train = np.load('D:/Bitcamp/Project/Frontalization/Imagenius/Numpy/korean_lux_x.npy') # Side face

print(X_train.shape)

X_train = X_train.reshape(15487, 49152)

X_train = X_train[0 : 100,  :  ]

np.savetxt('D:/img.list', X_train)