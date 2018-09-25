import sys
from data_generator import generateData
from data_generator import sin2PiX
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *


def train(numOfTraningDataPoints, orderOfPolynomial, sigmaOfNoise, lnOfLambda):
    """A train function you can customise"""
    print(f'Least squares regularization: N ={numOfTraningDataPoints}, M = {orderOfPolynomial}, sigma = {sigmaOfNoise}, ln(lambda) = {lnOfLambda} is plotting...')
    data = generateData(N=numOfTraningDataPoints, sigma=sigmaOfNoise)
    vectorX_T = data['xArray']
    vectorT_T = data['yArray']

    # Plot the training data points
    # Plot('x', 't', '', data={'x': vectorX_T, 'y': vectorT_T})
    plot(vectorX_T, vectorT_T, 'ro')
    xlabel('x')
    ylabel('t')
    ylim(bottom=-1.2, top=1.2)
    title(f'$t = \\sin(2\\pi x)$\n'
        f'带惩罚项的最小二乘法拟合 \n $N = {numOfTraningDataPoints}, '
        f'M = {orderOfPolynomial}, \\sigma = {sigmaOfNoise}, \\ln\\lambda = {lnOfLambda}$')

    # Get Vandermonde matrix X, see equation (8)
    matrixX = vander(vectorX_T, orderOfPolynomial+1, True)
    
    # Get the transpose of w_ML, see equation (16)
    w_ML_T = matmul(
        matmul(vectorT_T, matrixX),
        inv(
            matmul(transpose(matrixX), matrixX) +
            exp(lnOfLambda)*identity(orderOfPolynomial + 1)
        ),
    )

    # Print the solution for polynomial coefficients to file
    with open(f'../training_results/least-squares-regularization-{numOfTraningDataPoints}-{orderOfPolynomial}.txt', 'w+') as training_results:
        training_results.write(f'[w_0 w_1 ... w_{orderOfPolynomial}] = \n\t' + str(w_ML_T) + '\n\n')

    # Generate shorter intervals than vectorX_T
    vectorFittingX = arange(-2.0, 2.1, 0.000001)
    matrixFittingX = vander(vectorFittingX, orderOfPolynomial+1, True)
    # Plot the fitting curve, see equation (2)
    vectorY = transpose(matmul(matrixFittingX, transpose(w_ML_T)))
    plot(vectorFittingX, vectorY, 'g')
    # Plot sin(2 * pi * x)
    vector2PiX = array(list(map(sin2PiX, vectorFittingX)))
    plot(vectorFittingX, vector2PiX, 'y')

    # Save to /images
    savefig(f'../images/least-squares-regularization-{numOfTraningDataPoints}-{orderOfPolynomial}.png', bbox_inches='tight')
    close()


# Run training

# Case 1
train(numOfTraningDataPoints=40, orderOfPolynomial=10, sigmaOfNoise=0.2, lnOfLambda=-18)
# Case 2
train(numOfTraningDataPoints=40, orderOfPolynomial=20, sigmaOfNoise=0.2, lnOfLambda=-18)
# Case 3
train(numOfTraningDataPoints=40, orderOfPolynomial=39, sigmaOfNoise=0.2, lnOfLambda=-18)
# Case 4
train(numOfTraningDataPoints=20, orderOfPolynomial=19, sigmaOfNoise=0.2, lnOfLambda=-18)
