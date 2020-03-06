

#coding up the ips evaluator 
def action_match(pred, y, k):
    """Takes in two arrays, pred: the predicted action from target policy
    and y: the actual action in the test data.
    And returns: a matrix of actions (as key) and index (binary) """
    N = len(y)
    match = np.zeros( (N, k) )
    for i in range(N):
        if pred[i] == y[i]:
            match[i][y[i]] = 1

    return match

def ips_value(dic, prop):
    """Takes matrix of matches dic, and propensities matrix prop.
    Returns value of a target policy"""
    
    value = 0
    sample_used = sum(sum(dic))
    M = prop.shape[0]
    
    for i in range(M):
        
        prop_arr = list(dic[i])
        #print(prop_arr)
        if 1 in prop_arr:
            key  = prop_arr.index(1)
            #print(key)
            #print(i)
            try:
                if prop[i][key] < 0.1:#including clipping
                     value += 1. / 0.1
                else:
                    value += 1. / prop[i][key]
            except IndexError:
                print('No propensity for the action chosen, inspect action', key)
            #print(prop[i][key])
        else:
            continue
    return sample_used, value / M
