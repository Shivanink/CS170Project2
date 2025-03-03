import random
import numpy as np


#code stub
def leave_one_out_cross_validation(data, current_set, feature_to_add):
    accuracy = random.random() #we are replacing this code with the real code later
    return accuracy


def leave_one_out_cross_validation1(data, current_set, feature_to_add):
    
    return accuracy

    
      
accuracy = leave_one_out_cross_validation(None, None, None)



def forwardSelection(data): #data is the data set youre intaking
    size = data.shape #get num of rows and column in dataset
    dsize = size[1] #get number of columns (where attributes are) #IS LABEL AT 0 INDEX??

    current_set_of_features = [] #initialize empty list so we can keep track of items, Q: does it need to be set or list

    for i in range(1, dsize): #first column is just label so it doesn't count, go up to last column
        print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = None #should this be a set or is it jsut one val Q
        best_so_far_accuracy = 0

        for k in range(1, dsize): #these nested loops helps us traverse through the search space  
            if k not in current_set_of_features:  #dont add duplicate features, make sure each feature is added only once
                print(f"--Consider adding the {k} feature")
                accuracy = leave_one_out_cross_validation(data,current_set_of_features, k) #we are looking to remember highest num (IS IT K OR K+1)
               
                #get max
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k #k feature gave us this accuracy so we want to add it
                    
        current_set_of_features.append(feature_to_add_at_this_level) #add new feature to current set since we chose it
        print(f"On level {i} i added feature {feature_to_add_at_this_level} to current set")

def backwardSelection(data):
    size = data.shape
    dsize = size[1]

    #start with list of all features
    current_set_of_features =[]
    for ft in range(1, dsize):
        current_set_of_features.append(ft)
    
    for i in range(1,dsize):
        print(f"On the {i}th level of the search tree")
        feature_to_delete_at_this_level = None
        worst_so_far_accuracy = 0

        for k in range(1, dsize):
            if k in current_set_of_features: #make sure the feature exists in the set before you can delete it
                print(f"--Consider removing the {k} feature")
                accuracy = leave_one_out_cross_validation(data,current_set_of_features, k) #calculate accuracy
                
                #get the lowest accuracy
                if accuracy < worst_so_far_accuracy:
                    worst_so_far_accuracy = accuracy
                    feature_to_delete_at_this_level = k
            
        current_set_of_features.remove(feature_to_delete_at_this_level) #remove feature with the least accuracy
        print(f"On level {i} i removed feature {feature_to_delete_at_this_level} from the current set")




if __name__ == "__main__":
    data = np.loadtxt('CS170_Small_Data_000.txt')
    