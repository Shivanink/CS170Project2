import random
import numpy as np #better for arrays


#code stub
def leave_one_out_cross_validation1(data, current_set, feature_to_add):
    accuracy = random.random() #we are replacing this code with the real code later
    return accuracy

#Todo still:
# - add this to the search functions and make sure you can pass everything in (CHECK I THINK)
# - make sure to delete features youre not using. u can do this by setting them to 0
def leave_one_out_cross_validation(data, current_set, feature_to_add): #nearest neighbor, cs170 demo function in video
    #first column -> class     
    size = data.shape
    dsizecol = size[0]

    dcopy = np.copy(data)
    # dcopy = data.tonumpy()
    for i in range(1, dcopy.shape[1]):
        if i not in current_set and i != feature_to_add:
            dcopy[:,i]= 0



    number_correctly_classified = 0
    
    for i in range(dsizecol): #loop through each row
        object_to_classify = dcopy[i, 1:] #data that includes everything but the class/label column
        label_object_to_classify = dcopy[i, 0] #only the first label column

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = None
        

        for k in range(dsizecol):
            if k != i: #dont compare itself to itself
                #print(f"Ask if {i} is nearest neighbour with {k}")
                distance = (object_to_classify - dcopy[k,1:])**2 #square the difference
                distance = np.sqrt(np.sum(distance)) #square     root the sum of distance

                if distance < nearest_neighbor_distance: #find the smallest distance   
                    nearest_neighbor_distance = distance #distance from neighbor (smallest distance)    
                    nearest_neighbor_location = k #store index of nearest neighbor
                    nearest_neighbor_label = dcopy[nearest_neighbor_location, 0] #class of neighbor
        #print(f"Object{i} is class {label_object_to_classify}")
        #print (f"Its nearest neighbor is {nearest_neighbor_location} which is in class {nearest_neighbor_label}")

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    
    accuracy = number_correctly_classified/(dsizecol)



        #print(f"Looping over i, at the {i} location")
        #print(f"The {i}th object is in class {label_object_to_classify}")



    return accuracy

    



def forwardSelection(data): #data is the data set youre intaking
    size = data.shape #get num of rows and column in dataset
    dsize = size[1] #get number of columns (where attributes are) #IS LABEL AT 0 INDEX??

    current_set_of_features = [] #initialize empty list so we can keep track of items, Q: does it need to be set or list
    global_accuracy = 0 #best overall accuracy

    for i in range(1, dsize): #first column is just label so it doesn't count, go up to last column
        print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = None #should this be a set or is it jsut one val Q
        best_so_far_accuracy = 0

        dcopy = np.copy(data) #MAYBE DELETE

        for k in range(1, dsize): #these nested loops helps us traverse through the search space  
            if k not in current_set_of_features:  #dont add duplicate features, make sure each feature is added only once
                temp_set = np.copy(dcopy)
                
                accuracy = leave_one_out_cross_validation(data,current_set_of_features, k) #we are looking to remember highest num (IS IT K OR K+1)
                print(f"--Consider adding the {k} feature. Accuracy: {accuracy:.4f}")
               
                #get max (local accuracy) -> getting the best accuracy in k
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k #k feature gave us this accuracy so we want to add it

        
        #get the highest accuracy of all calculate accuracies so far
        if global_accuracy < best_so_far_accuracy:
            global_accuracy = best_so_far_accuracy
            current_set_of_features.append(feature_to_add_at_this_level) #add new feature to current set since we chose it
            print(f"On level {i} i added feature {feature_to_add_at_this_level} to current set. Best so far accuracy: {best_so_far_accuracy:.4f}")
        else:
            print(f"Warning, Accuracy has decreased! Continuing search in case of local maxima") #ask if this is right

    print(f"Final Accuracy:{global_accuracy:.4f}")
    print(f"Final Selected Features: {current_set_of_features}")

def backwardElimination(data):
    size = data.shape
    dsize = size[1]

    #start with list of all features
    current_set_of_features =[]
    global_accuracy = leave_one_out_cross_validation(data, current_set_of_features, None)
    for ft in range(1, dsize):
        current_set_of_features.append(ft)
    
    for i in range(1,dsize):
        print(f"On the {i}th level of the search tree")
        feature_to_delete_at_this_level = None
        best_so_far_accuracy = global_accuracy #float('inf')

        for k in range(1, dsize):
            if k in current_set_of_features: #make sure the feature exists in the set before you can delete it
                temp_set = current_set_of_features.copy()
                temp_set.remove(k)

                accuracy = leave_one_out_cross_validation(data,temp_set, None) #calculate accuracy
                print(f"--Consider removing the {k} feature.Accuracy: {accuracy:.4f} ")
                
                #get the lowest accuracy
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_delete_at_this_level = k
        

        if feature_to_delete_at_this_level is not None and global_accuracy <= best_so_far_accuracy:    
            global_accuracy = best_so_far_accuracy
            current_set_of_features.remove(feature_to_delete_at_this_level) #remove feature with the least accuracy
            print(f"On level {i} i removed feature {feature_to_delete_at_this_level} from the current set. Best so far accuracy: {best_so_far_accuracy:.4f}")
    

        #get the highest accuracy of all calculate accuracies so far
        if global_accuracy > best_so_far_accuracy:
            print(f"Warning, Accuracy has decreased! Continuing search in case of local maxima")

        if len(current_set_of_features) <=1:
            break
    print(f"Final Accuracy:{global_accuracy:.4f}")
    print(f"Final Selected Features: {current_set_of_features}")




if __name__ == "__main__":
    print("Welcome to Shivani Nandakumar's Feature Selection Algorithm!")
    dataSet = input("Type in the name of the file to test: ")
    print("Type the number of the algorithm you want to run.")
    print(" (1) Forward Selection")
    print(" (2) Backward Elimination")
    option = input("Option: ")
    data = np.loadtxt(dataSet)
    #forwardSelection(data)

    if option == "1":
        forwardSelection(data)
    elif option == "2":
        backwardElimination(data)

    