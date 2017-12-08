import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    phi_y = np.mean(category)
    matrix_0 = matrix[category == 0, :]
    log_phi_k0 = np.log((1.0 + np.sum(matrix_0, axis=0)) / (np.sum(matrix_0) + N))
    matrix_1 = matrix[category == 1, :]
    log_phi_k1 = np.log((1.0 + np.sum(matrix_1, axis=0)) / (np.sum(matrix_1) + N))
    state = {'phi_y': phi_y, 
             'log_phi_k0': log_phi_k0, 
             'log_phi_k1': log_phi_k1}
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    log_prod_0 = np.dot(matrix, state['log_phi_k0']) + np.log(1 - state['phi_y'])
    log_prod_1 = np.dot(matrix, state['log_phi_k1']) + np.log(state['phi_y'])
    output = 1 * (log_prod_0 < log_prod_1)
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error
    return error
    
def find_best_tokens(state, tokenlist, n = 5):
    diff = state['log_phi_k1'] - state['log_phi_k0']
    ind = np.argpartition(diff, -n)[-n:]
    ind = ind[np.argsort(diff[ind])][::-1]
    best_tokens = np.array(tokenlist)[ind].tolist()
    return best_tokens

def main():
    
    # Question a
    
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    error = evaluate(output, testCategory)
    
    # Question b
    
    best_tokens = find_best_tokens(state, tokenlist)
    print best_tokens
    
    # Question c
    
    # Get errors for different sizes 
    print 'Training for different sizes'
    sizes = [50, 100, 200, 400, 800, 1400]
    errors = []
    for size in sizes:
        strainM, _, strainC = readMatrix('MATRIX.TRAIN.{}'.format(size))
        sstate = nb_train(strainM, strainC)
        soutput = nb_test(testMatrix, sstate)
        errors.append(evaluate(soutput, testCategory))
    errors.append(error)
    sizes.append(int(trainMatrix.shape[0]))
    # Plot
    plt.figure()
    plt.plot(sizes, errors, label = 'Errors')
    plt.plot(sizes, errors, '+')
    plt.xlabel('Training set size')
    plt.ylabel('Test set error with Naive Bayes')
    plt.savefig('ps2q6c.png', dpi = 1200)
    

if __name__ == '__main__':
    main()
