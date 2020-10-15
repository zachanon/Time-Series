import numpy as np
import pandas as pds

def batch_autoregressive_energydata(batch_dim, order):
    
    data = pds.read_csv("data/energydata_complete.csv").values
    
    #timestamps are in regular 10 minute intervals, so unnescessary
    data = data[:,1:]
    
    numdata = data.shape[0]
    order = order
    dim = data.shape[-1]
    numsamples = numdata-order+1

    sequenced = np.zeros([numsamples, order, dim])
    for n in range(0, numsamples):
        begin = n
        end = n+order

        sequenced[n]= data[begin:end]
        
    numbatches = round(numsamples/batch_dim)
    
    batched = np.reshape(sequenced[:batch_dim*numbatches], [numbatches, batch_dim, order, dim])
    
    return batched

def prepLinearBatches(data):
    
    samples, order, dim = data.shape
    
    return np.reshape(data, [samples, order*dim])