import numpy as np
from scipy import stats

class KNNClassifier:
    '''
    K-NN classifier class
    Supports fit, predict and predict_proba function
    '''
    
    #class For K-NN Classifier
    
    def __init__(self, K = 5):
        '''
        Initialize K value here

        Input:
            K: int, default = 5
                No. of nearest neighbors
        '''
        
        #initializing the class varaibles
        self.K = K
        self.x_train = []
        self.y_train = []
        self.classes = []
        self.n_datapoints = 0
        self.n_classes = 0
    
    def fit(self, x_train, y_train):
        '''
        Fit function to load the train data into RAM

        Input:
            x_train: numpy array of shape (n_points, n_features)
            y_train: numpy array of shape (n_points,)
        '''
        
        #fitting the data into memory
        self.x_train = x_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)
        self.n_datapoints = len(x_train)
        print(f'KNeighborsClassifier, K = {self.K}')
        
    def predict(self, x_test):
        '''
        Function to predict the class label for given query points

        Input:
            x_test: numpy array of shape (n_points, n_features)

        Returns:
            y_test: numpy array of shape (n_points,)
        '''
        
        y_pred = np.zeros(len(x_test))

        #we will need to iterate through every query point
        for i, x in enumerate(x_test):

            #we will store the distance of query point from each train point and the class label for that training point
            distance_and_neighbors = []

            #iterating through each of the training point to calculate distance between itself and query point
            for x_tr, y_tr in zip(self.x_train, self.y_train):
                
                d = self.eucl_distance(x, x_tr)
                distance_and_neighbors.append((d, y_tr))

            #sorting the distances and neighbors in ascending order of distance
            distance_and_neighbors = sorted(distance_and_neighbors, key = lambda k: k[0])
            neighbors = [ele[1] for ele in distance_and_neighbors[:self.K]]
            y_pred[i] = stats.mode(neighbors)[0][0]
            
        return y_pred

    def predict_proba(self, x_test):
        '''
        Function to predict the class label for given query points

        Input:
            x_test: numpy array of shape (n_points, n_features)

        Returns:
            y_test_proba: numpy array of shape (n_points, n_classes)
        '''

        y_pred_proba = np.zeros((self.n_datapoints, self.n_classes))
        
        #we will need to iterate through every query point
        for i, x in enumerate(x_test):

            #we will store the distance of query point from each train point
            #and the class label for that training point
            distance_and_neighbors = []

            #iterating through each of the training point to calculate distance between itself and query point
            for x_tr, y_tr in zip(self.x_train, self.y_train):
                
                d = self.eucl_distance(x, x_tr)
                distance_and_neighbors.append((d, y_tr))

            #sorting the distances and neighbors in ascending order of distance
            distance_and_neighbors = sorted(distance_and_neighbors, key = lambda k: k[0])
            
            #fetching the neighboring class labels
            neighbors = np.array([ele[1] for ele in distance_and_neighbors[:self.K]])
            
            #now we will calculate the probability of each class label in nearest neighbors
            for j, class_label in enumerate(self.classes):
                y_pred_proba[i,j] = len(neighbors[neighbors == class_label]) / self.K
                
        return y_pred_proba
                  
    def eucl_distance(self, x1, x2):
        '''
        Function to calculate Euclidean distances between two vectors

        Input:
            x1, x2: input vectors
        Returns:
            Scaler Euclidean Distance
        '''
        
        return np.linalg.norm(x1-x2)
