#general test function
from pandas.io.html import _importers
from tensorflow.python.ops.gen_array_ops import matrix_set_diag
from networkx.algorithms.centrality import eigenvector
def test_function(function, test_cases, test_cases_answers):
    """Runs function through test cases"""
    passed = True
    
    for (case, answer) in zip(test_cases, test_cases_answers):
        output = function(case)
        if output != answer: 
            print('Your output: ', output, '\t Expected output: ', answer)
            passed = False
    
    if passed: 
        print('Test passed.')
    else: 
        print('Test failed.')

#1.average 
def average(array):
    sum = 0
    
    for i in range(len(array)):
        sum+=array[i]
    
    return sum/len(array)

'''
other solutions:

def average(array):
    avg = sum(array)/len(array)
    return avg

def average(array):
    return np.average(array)

def test_function(average, test_cases_average, test_cases_answers_average):
    
    for i in test_cases_answers_average:
        average = average(test_cases_average[i])
        if average == test_cases_answers_average[i]:
            print(1)
            return True
        else:
            print(0)
            return False 
'''

test_cases_average = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [0.21780088, 0.4118428,  0.62573827, 0.78184048, 0.96162729, 0.21242742, 0.06843751, 0.70688805, 0.98459559, 0.1153083 ]]

test_cases_answers_average = [1, 5.5, 0.508650659]

test_function(average, test_cases_average, test_cases_answers_average)


#############################################################
#2.extract
def extract_username(email):
    list_1 = email.split('@')
    return list_1[0]


'''
another solution:

'''


test_cases_extract_username = ['john-smith@gmail.com',
              'this.is.the.username@abcdefghijklmnopqrstuvwxyz',
              'invalid-email']

test_cases_answers_extract_username = ['john-smith', 'this.is.the.username', 'invalid-email']

test_function(extract_username, test_cases_extract_username, test_cases_answers_extract_username) 

#############################################################
#3.& 4. numpy vector x
import numpy as np
import random 

x = np.zeros([1,10])
for i in range(10):
    x[0][i] = random.randint(10,15)
print(x)

#max
print(max(x[0]))

#max position
print(np.where(x[0]==max(x[0])))
a = np.where(x[0]==max(x[0])) #a is a tuple

#############################################################
#5.numpy vector y ("1"s)
y = np.ones([1,10])
print(y)

#############################################################
#6. add x and y (numpy ufuncs)
answer = np.add(x,y)
print(answer)

#############################################################
#7.matrix
matrix = np.array([[5,10,15],[2,13,23]])
print(matrix)

#############################################################
#8.transpose
matrix_T = matrix.T 
print(matrix_T)

#############################################################
#9.multiplication
matrix_square = np.matmul(matrix,matrix_T) 
print(matrix_square)

#det
det = np.linalg.det(matrix_square)
print(det)

#inv
inv = np.linalg.inv(matrix_square)
print(inv)

#eigens
eigenvalues ,eigenvectors = np.linalg.eig(matrix_square)
print('eigenvalues: ', eigenvalues)
print('eigenvectors: ', eigenvectors)

#check eig
result_1 = np.dot(matrix_square,[eigenvectors[0][0],eigenvectors[1][0]])
result_2 = np.dot(eigenvalues[0],[eigenvectors[0][0],eigenvectors[1][0]])
print(result_1,result_2)


#############################################################
#10. Create a 3-element vector with integers of your choice.
v = np.zeros(3)
print(v) 

#############################################################
#11. multiply matrix and v
answer = np.dot(matrix, v)
print(answer)

#############################################################
#12. flatten
flatten = matrix.flatten()
print(flatten)

##############################################################
# 13. concatenate
matrix1 = [[1, 2, 3, 4, 5], 
           [6, 7, 8, 9, 10], 
           [11, 12, 13, 14, 15]]

matrix2 = [[16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25],
           [26, 27, 28, 29, 30]]
mat1 = np.array(matrix1)
mat2 = np.array(matrix2)

ans_0 = np.concatenate((mat1, mat2), axis=0)
print(ans_0)

ans_1 = np.concatenate((mat1, mat2), axis=1)
print(ans_1)

#############################################################

#statistics

#randint()
v = np.random.randint(1,20,[1,30])
print(v)

#median, std 
md = np.median(v)
std = np.std(v)

#random sampling
list = np.arange(10) 
print(list)

sample = np.random.choice(list, 5)
print(sample)

sample = np.random.choice(list, 5, replace = False) #without replacement 
print(sample)

#############################################################
#Maximum A Posteriori

toss = 200
total = 0
head = 0
theta = 0.5
sample = np.array([1,1,1,1,1,1,1,1,0,0])
res = np.zeros(toss)

for i in range(toss):
    j=i+1
    print("toss",j)
    print("last theta: ",theta)
    if np.random.choice(sample)==1:
        head+=1
        res[i]=1
    total+=1
    theta = head/total
    print("new theta: ",theta)
    print("---------")

print(res)

################
#DONE