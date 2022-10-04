# import os
#
# # assign directory
# directory = 'delete'
#
# # iterate over files in
# # that directory
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     name = filename.split('.')[-2]
#     print(name)
#     # checking if it is a file
#     print(f)
#     print('***************')

# import pandas as pd
# import pickle
# import numpy as np
#
# face_embedding = pickle.load(open('face_embeddings.pkl', 'rb'))
#
# print(face_embedding.shape)
# for i, rows in face_embedding.iterrows():
#     print(rows['Name'])
#     print(rows['Embedding'])
# # print(face_embedding['Embedding'])

# li = ['jhiya', 'chot', 'lagi', 'kanha']
# lis = ['chitwan', 'ki', 'rahiya', 'bhula']
#
# compare = []
#
# for i in zip(li, lis):
#     compare.append(i)
#
# print(compare[1][0])

li = ['jhiya', 'chot', 'lagi', 'kanha']
# for idx, value in enumerate(li):
#     print(idx)
#     print(value)

value = 'chot'
if value in li:
    print("chitwan")