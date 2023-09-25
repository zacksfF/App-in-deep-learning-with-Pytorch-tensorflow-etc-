import numpy as np

def cross_entropy(prediction, targets, epsilon=1e-10):
    prediction = np.clip(prediction, epsilon, 1. - epsilon)
    N = prediction.shape[0]
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5)))/N
    return ce_loss

predictions = np.array([[0.25, 0.25, 0.25, 0.25],
                        [0.01, 0.01, 0.01, 0.96]])
targets = np.array([[0,0,0,1],
                    [0,0,0,1]])

cross_entropy_loss = cross_entropy(predictions, targets)
print("Cross entropy loss is :"+ str(cross_entropy_loss))
 









