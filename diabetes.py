#!/usr/bin/env python
# coding: utf-8

# In[12]:


from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy
import tensorflow as tf

numpy.random.seed(3)
tf.set_random_seed(3)

df_pre = pd.read_csv('Diabet_binary5050.csv')
df = df_pre.sample(frac=0.1) #전체데이터 중 10%만 랜덤하게 선정하여 사용함
data = df.values
X = data[:, 1:9] # 속성
Y = data[:, 0] #당뇨병클래스(1=당뇨병, 0=당뇨병아님)

colormap=plt.cm.gist_heat
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5,cmap=colormap,linecolor='white',annot =True)
plt.show()

plt.figure(2)
grid=sns.FacetGrid(df,col='Diabetes_binary')
grid.map(plt.hist,'Income',bins=10)
plt.show()

model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

hist = model.fit(X,Y,epochs=200,batch_size=10)
print("\n Accuracy:%.4f"%(model.evaluate(X,Y)[1]))

#빨간색 손실,파란색 예측 표로 나타내기
y_vloss=hist.history['loss'] #테스트셋으로 실험결과 오차값 저장

y_acc=hist.history['acc']        #학습셋으로 측정한 정확도값 저장

x_len=numpy.arange(len(y_acc))
plt.plot(x_len,y_vloss,"o",c="red",markersize=3)
plt.plot(x_len,y_acc,"o",c="blue",markersize=3)
plt.ylim([0.5,0.8])
plt.show



# In[3]:


# Get the figure and the axes
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=False, figsize=(10, 5))

# 모델의 정확도를 그립니다.
ax0.plot(history.history['acc']) 
ax0.set(title='model accuracy', xlabel='epoch', ylabel='accuracy')

# 모델의 오차를 그립니다.
ax1.plot(history.history['loss'])
ax1.set(title='model loss', xlabel='epoch', ylabel='loss')



# In[5]:


# 가상의 환자 데이터 입력
patient_1 = numpy.array([[1,1,1,40,1,1,1,1]])

# 모델로 예측하기
prediction = model.predict(patient_1)

# 예측결과 출력하기
print(prediction*100)


# In[ ]:





# In[ ]:




