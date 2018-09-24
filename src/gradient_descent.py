import sys
from data_generator import generateData
from data_generator import sin2PiX
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *


def train(numOfTraningDataPoints, orderOfPolynomial, sigmaOfNoise, lnOfLambda, learningRate, precision):
    """A train function you can customise (use gradient descient)"""
    print(f'Gradient descient: N ={numOfTraningDataPoints}, '
        f'M = {orderOfPolynomial}, '
        f'sigma = {sigmaOfNoise} is plotting...')
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

    # Gradient function
    def gradient(w):
        return matmul(
            transpose(matrixX),
            matmul(matrixX, w) - vectorT_T.reshape(-1, 1)
            ) + exp(lnOfLambda) * w

    # Initialize the polynomial with ones
    cur_w = ones((orderOfPolynomial + 1, 1))
    previous_step_size = 1
    iters = 0
    while previous_step_size > precision:
        learning = gradient(cur_w) * learningRate
        cur_w -= learning
        previous_step_size = linalg.norm(learning)
        print('Current learning: ', previous_step_size)
        iters += 1


    title(f't = sin(2$\pi x$)\n'
        f'梯度下降法拟合 \n N = {numOfTraningDataPoints},'
        f'M = {orderOfPolynomial}, $\sigma$ = {sigmaOfNoise}, 学习率 $\\alpha$ = {learningRate}, 截止步长 = {precision}\n'
        f'迭代次数: {iters} 次')

    # Print the solution for polynomial coefficients to file
    with open(f'training_results/gradient-descent-{numOfTraningDataPoints}-{orderOfPolynomial}.txt', 'w+') as training_results:
        training_results.write(f'[w_0 w_1 ... w_{orderOfPolynomial}] = \n\t' + str(transpose(cur_w).reshape(-1)) + '\n\n')

    # Generate shorter intervals than vectorX_T
    vectorFittingX = arange(-2.0, 2.1, 0.000001)
    matrixFittingX = vander(vectorFittingX, orderOfPolynomial+1, True)
    # Plot the fitting curve, see equation (2)
    vectorY = transpose(matmul(matrixFittingX, cur_w)).reshape(-1)
    plot(vectorFittingX, vectorY, 'g')
    # Plot sin(2 * pi * x)
    vector2PiX = array(list(map(sin2PiX, vectorFittingX)))
    plot(vectorFittingX, vector2PiX, 'y')

    # Save to /images
    savefig(f'images/gradient-descent-{numOfTraningDataPoints}-{orderOfPolynomial}.png', bbox_inches='tight')
    close()
    print(f'Done! iteration times: {iters}')


# Run training

# Case 1
train(numOfTraningDataPoints=4, orderOfPolynomial=2, sigmaOfNoise=0.2, lnOfLambda=-5, learningRate=0.01, precision=1e-10)

# Case 2
train(numOfTraningDataPoints=10, orderOfPolynomial=3, sigmaOfNoise=0.2, lnOfLambda=-5, learningRate=0.01, precision=1e-10)

# Case 2
train(numOfTraningDataPoints=10, orderOfPolynomial=9, sigmaOfNoise=0.2, lnOfLambda=-5, learningRate=0.000002, precision=1e-6)