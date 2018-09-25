import sys
from data_generator import generateData
from data_generator import sin2PiX
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *


def train(numOfTraningDataPoints, orderOfPolynomial, sigmaOfNoise, lnOfLambda, precision):
    """A train function you can customise (use conjugate gradient)"""
    print(f'Conjugate gradient: N ={numOfTraningDataPoints}, M = {orderOfPolynomial}, sigma = {sigmaOfNoise} is plotting...')
    data = generateData(N=numOfTraningDataPoints, sigma=sigmaOfNoise)
    vectorX_T = data['xArray']
    vectorT_T = data['yArray']

    # Plot the training data points
    # Plot('x', 't', '', data={'x': vectorX_T, 'y': vectorT_T})
    plot(vectorX_T, vectorT_T, 'ro')
    xlabel('x')
    ylabel('t')
    ylim(bottom=-1.2, top=1.2)

    # Get Vandermonde matrix X, see equation (8)
    matrixX = vander(vectorX_T, orderOfPolynomial+1, True)

    # Get matrix B, see equation (20)
    matrixB = matmul(transpose(matrixX), matrixX) + exp(lnOfLambda) * identity(orderOfPolynomial + 1)
    # Initialize variables
    w = zeros((orderOfPolynomial+1, 1))
    r = matmul(transpose(matrixX), vectorT_T.reshape(-1, 1)) - matmul(matrixB, w)
    p = r
    k = 0
    # Begin iterating
    while True:
        alpha = matmul(transpose(r), r) / matmul(matmul(transpose(p), matrixB), p)
        new_w = w + alpha * p
        new_r = r - alpha * matmul(matrixB, p)
        # Exit if new_r is small enough
        if(linalg.norm(new_r) < precision):
            w = new_w
            break
        beta = matmul(transpose(new_r), new_r) / matmul(transpose(r), r)
        new_p = new_r + beta * p
        
        w = new_w
        r = new_r
        p = new_p
        k = k+1

    # Print the solution for polynomial coefficients to file
    with open(f'../training_results/conjugate-gradient-{numOfTraningDataPoints}-{orderOfPolynomial}.txt', 'w+') as training_results:
        training_results.write(f'[w_0 w_1 ... w_{orderOfPolynomial}] = \n\t' + str(w.reshape(-1)) + '\n\n')

    title(f't = sin(2$\\pi x$)\n'
        f'共轭梯度法 - $N = {numOfTraningDataPoints}, '
        f'M = {orderOfPolynomial}, \\sigma = {sigmaOfNoise}, \\ln\\lambda = {lnOfLambda}, precision = {precision}$\n'
        f'迭代次数: {k} 次')
    # Generate shorter intervals than vectorX_T
    vectorFittingX = arange(-2.0, 2.1, 0.000001)
    matrixFittingX = vander(vectorFittingX, orderOfPolynomial+1, True)
    # Plot the fitting curve, see equation (2)
    vectorY = transpose(matmul(matrixFittingX, w.reshape(-1)))
    plot(vectorFittingX, vectorY, 'g')
    # Plot sin(2 * pi * x)
    vector2PiX = array(list(map(sin2PiX, vectorFittingX)))
    plot(vectorFittingX, vector2PiX, 'y')

    # Save to /images
    savefig(f'../images/conjugate-gradient-{numOfTraningDataPoints}-{orderOfPolynomial}.png', bbox_inches='tight')
    close()


# Run training

# Case 1
train(numOfTraningDataPoints=4, orderOfPolynomial=2, sigmaOfNoise=0.2, lnOfLambda=-5, precision=1e-10)

# Case 2
train(numOfTraningDataPoints=10, orderOfPolynomial=3, sigmaOfNoise=0.2, lnOfLambda=-5, precision=1e-10)

# Case 3
train(numOfTraningDataPoints=10, orderOfPolynomial=9, sigmaOfNoise=0.2, lnOfLambda=-5, precision=1e-6)