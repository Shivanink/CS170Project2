import random
import numpy as np

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    accuracy = random.random() #we are replacing this code with the real code later
    return accuracy

    
      
accuracy = leave_one_out_cross_validation(None, None, None)



def feature_search_demo(data): #data is the data set youre intaking
    dsize = data.shape() #get num of rows and column in dataset
    dsize = dsize[1] #get number of columns (where attributes are)
    for i in range(1, dsize - 1): #first column is just label so it doesn't count
        print(f"On the {str(i)}th level of the search tree")
        for k in range(1, dsize-1):
            print(f"On the {str(k)}th level of the search tree")



    