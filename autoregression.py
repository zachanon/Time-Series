import numpy as np

class Autoregression(object):
    
    def __init__(self, order, dim_x, initializer='random_normal'):
        """
        A callable model object that implements the autoregression function.
        
        param:
            order (Int): the order of the autoregression, how many timesteps to go back
            dim_x (Int): the number of features in the input
            initializer (String): the initializer for the weights
            
        """
        
        assert(order > 0),"Order must be greater than 0"
        self.order = order
        
        if(initializer=='random_normal'):
            self.weights = np.random.randn(order+1, dim_x)
            
        else:
            raise ValueError("Initializer " + initializer + " not implemented.")
            
    def __call__(self, x):
        """
        param:
            x (Array or Tensor): the input variable to regress
                        shape: [order, num_features]
        returns:
            y (Float): the resulting value calculated from the model
        """
        #add an extra dim to x for the bias
        x = np.reshape(np.append(x, 1), [x.shape[0],x.shape[1]+1])
        
        #weights shape [order, (dim_x=num_features)+1]
        #x shape [order, num_features+1]
        # y shape []
        y = np.einsum('ij,ij->', self.weights, x)
        
        return y