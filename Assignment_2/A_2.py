#data man and vis

#1.import pack
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import cv2

# hint: never overwrite your original dataset because you never know when you'll need it again.


#2.data man
df = pd.read_csv('/Users/stevenchan/Documents/TECHX_2019/Self/Assignments/Assignment2/dataset/ramen-ratings.csv')
df.fillna(value = 'no') #replace NaN
df['Top Ten'] = df['Top Ten'].str.split('#').str[0] #replace '#' in Top Ten

###########################

df['Stars'] = pd.to_numeric(df['Stars'])# cast the ratings to numeric numbers from strings 
avg = df.groupby(['Country'])['Stars'].mean()# rank all countries based on stars review
avg = avg.reset_index() # resets the column names to the origin ones 
avg = avg.sort_values('Stars')
#avg.plot.bar(x='Country', y='Stars')

###########################

df.tail() #defult 5 rows, tail(1)==1 row
brazil = df[df['Country'] == 'Brazil'] # find the Brazil rows in Country 

###########################

brazil_packs = brazil[(brazil['Style'] == 'Pack')] # select Pack style from Brazil
to_try = brazil_packs.loc[brazil_packs['Stars'].idxmax()] #***

###########################

countries = ['Japan', 'Malaysia', 'South Korea', 'Singapore']
print(df[(df['Country'].isin(countries)) & (df['Top Ten'] != 'no')]) # include these countries only
# cannot use 'and', but'&'  ==> 1 & 0 = 0; 1 & 1 = 1

###########################

# In a single day finishes ALL the ramens above 3.0 (inclusive) stars, that are NOT in a 'Pack'. (Yes, he eats a lot). 
# Change all of the 'Tries' entries of these rows to True.
mask = (df['Country'] == 'Singapore') & (df['Stars'] >= 3.0) & (df['Style'] != 'Pack') # all 
df.loc[mask, 'Tried'] = True # choose all the mask rows and make a new 'Tried' column
print(df[df['Country'] == 'Singapore'])  # can print all columns of this one row

        # similarly, loc[] can do this:
df.loc[df['Top Ten'] == '\n', 'Top Ten'] = 'no' # replace '\n' by 'no'

# To show all the column info: 
#   df.columns.values     ==> return a array


'''The Pandas loc indexer can be used with DataFrames for two different use cases:
    a. Selecting rows by label/index
    b. Selecting rows with a boolean / conditional lookup'''

###########################

# make a new row
df = df.append(pd.Series([len(df) + 1, # Review #: Increment by 1 from the size of the dataset, 
           'Prima Taste', # Brand
           'Beef Noodle Soup', # Variety
           'Pack', # Style
           'Singapore', # Country
           5, # Stars
           'no', # Top Ten 
           True], index=df.columns), # Tried
           ignore_index=True) 

print(df[df['Variety'] == 'Beef Noodle Soup'])

df.drop(df[df['Tried'] == True].index, inplace=True) # drop True
df.drop('Tried', axis=1, inplace=True)
print(df)

###########################

style = df['Style'].value_counts()
print(style)
style.plot.pie() # count values in Style and make a pie chart

df.drop(df[df['Country'] == 'Singapore'].index, inplace=True)  # .index can show index, 
                                                               # returning pandas things: type(df[df['Country'] == 'Singapore'].index)
df.to_csv('./dataset/assignment-ramen-ratings.csv', index=False) #save file to_csv





######################################################
# data vis

dog = cv2.imread('/Users/stevenchan/Documents/TECHX_2019/Private/Assignment-Solutions/Assignment2/dataset/shiba-inu.jpg')
dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)



# =========== YOUR CODE HERE ======== 
plt.imshow(dog)
plt.show()

plt.figure(figsize=(100, 120)) #change size
plt.imshow(dog)
plt.show()

baby_dog = cv2.imread('./dataset/baby-dog.jpg')
baby_dog = cv2.cvtColor(baby_dog, cv2.COLOR_BGR2RGB)
star_boo = cv2.imread('./dataset/star-boo.jpg')
star_boo = cv2.cvtColor(star_boo, cv2.COLOR_BGR2RGB)
chihuahua = cv2.imread('./dataset/chihuahua.jpg')
chihuahua = cv2.cvtColor(chihuahua, cv2.COLOR_BGR2RGB)


# Create a subplot for each figure. 
plt.subplot(221) 
# subplot is equivalent but more general to ax1=plt.subplot(2, 2, 1)
plt.imshow(dog)
plt.subplot(222)
plt.imshow(baby_dog)
plt.subplot(223)
plt.imshow(star_boo)
plt.subplot(224)
plt.imshow(chihuahua)

''' Usage:
    subplot(nrows, ncols, index, **kwargs)
    subplot(pos, **kwargs)
    subplot(ax)
    '''
plt.show()