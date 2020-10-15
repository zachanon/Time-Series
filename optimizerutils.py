import numpy as np

def autoregressionNormalEquation(model, data):
    """
    Updates the weights of the input model using the Least Squares method
    
    param:
        model (Function): a callable autoregression model, compatible with the data
        data (Tensor): a tensor containing the data to optimize the model on
                size: [num_examples, num_features]
    
    """
    
    x = np.array(model.order)
    
    for i in range(0,data.shape[0]-model.order):
        for j in range(0, model.order):
            x[j] = i+j
        
        #model.weights shape [order, dim_x]
        #x shape [order, num_features=dim_x]
        #y_hat shape []
        y_hat = np.einsum('ij,ij->',model.weights, x) + model.bias
        
        cost = np.sqrt((y - y_hat)**2)