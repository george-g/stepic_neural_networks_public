import numpy as np

def cost_function(network, test_data, l1=0.0, l2=0.0, onehot=True,):
    c = 0 
        
    for example, y in test_data:
        if not onehot:
            y = np.eye(3, 1, k=-int(y))
        yhat = network.feedforward(example)
        c += np.sum((y - yhat)**2)
    
    c = c / len(test_data)
    
    #Добавляем регуляцию
    for w in network.weights:
        c += l1 * np.sum(abs(w)) +  l2 * np.sum(w**2) / 2
        
    return c  