import numpy as np

toss = 20
head = 0
total = 0
sample = [1,1,1,1,1,1,1,1,0,0]
theta = 0.5
P = theta**head*(1-theta)**(total-head)
a=2
b=2

for i in range(20):
    print("old theta:", theta)
    result = np.random.choice(sample)
    if result == 1:
        head+=1
    total+=1
    theta = (a+head-1)/(a+b+total-2)
    print("new theta:", theta)
    print("==========")

    #后验概率 MAP  最大似然 MLE
