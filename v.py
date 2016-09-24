import glob
import numpy as np
import pandas as pd
from scipy import linalg as la
from collections import OrderedDict
from six.moves import cPickle as pickle


'''############## Load Data ##############'''


def load_data():

    print('Loading data ...')

    data = OrderedDict()
    for file_ in glob.glob('*.csv'):

        if file_.split('.')[0] != 'estimates':

            data_ = OrderedDict()

            # Extract data from csv and load into vars x, y, z
            raw_data = np.genfromtxt(file_, delimiter=',', skip_header=True)
            data_['x'] = raw_data[:, 0].flatten()
            data_['y'] = raw_data[:, 1].flatten()

            # Create nested dicts store loss and model parameters
            data_['loss_ols'] = [0]
            data_['loss_irls'] = [0]
            data_['loss_lap'] = [0]
            data_['loss_gauss'] = [0]
            data_['b_ols'] = []
            data_['b_irls'] = []
            data_['b_lap'] = [np.array([0.1, 0.])]
            data_['b_gauss'] = [np.array([0.1, 0.])]
            data_['lmda_lap'] = [np.array([0.1, 0.])]
            data_['lmda_gauss'] = [np.array([0.1, 0.])]

            # Save to ordered dict
            data[file_.split('.')[0]] = data_

    files = data.keys()
    print('Data keys: %s' % files)

    return data, files


'''############## Ordinary Least Squares ##############'''


def run_OLS():

    print('Finding OLS estimates ...')

    for dataset in files:

        # Compute model parameters
        x = data[dataset]['x']
        x_ = np.stack((np.ones(shape=(len(x))), x))
        y = data[dataset]['y'].reshape(1, -1)
        b = la.inv(x_.dot(x_.T)).dot(x_.dot(y.T))
        data[dataset]['b_ols'].append(b)


'''############## Iteratively Reweighted Least Squares ##############'''


def run_IRLS(steps, k):

    print('Finding IRLS estimates ...')

    for step in range(steps):

        for dataset in files:

            # Load b, x, y
            x = data[dataset]['x']
            x = np.stack((np.ones(shape=(len(x))), x))
            y = data[dataset]['y'].reshape(1, -1)
            b = data[dataset]['b_ols'][-1]

            # Calculate dataset residuals
            resids = np.abs(y - b.T.dot(x)).flatten()

            # Calculate Huber-loss, construct weight-matrix (w)
            loss = 0
            n_ = len(resids)
            w = np.zeros(shape=(n_, n_))
            for i in range(n_):
                if resids[i] > k:
                    loss += 0.5 * resids[i] ** 2
                    w[i, i] = k / resids[i]
                else:
                    loss += k * resids[i] - 0.5 * k ** 2
                    w[i, i] = 1.
            data[dataset]['loss_irls'].append(loss / float(n_))

            # Update model parameters
            b = la.inv(x.dot(w.dot(x.T))).dot(x.dot(w.dot(y.T)))
            data[dataset]['b_irls'].append(b)

    # OLS residuals
    resids = np.array(())
    for dataset in files:
        x = data[dataset]['x']
        x = np.stack((np.ones(shape=(len(x))), x))
        y = data[dataset]['y'].reshape(1, -1)
        b = data[dataset]['b_ols'][-1]
        resids = np.append(resids, y - b.T.dot(x))

    return resids


'''############## Maximum Likelihood Estimators ##############'''


def neg_log_likelihood_gaussian(b, lmda, x, y):
    l = 0
    n_ = len(x)
    for i in range(n_):
        log_var_term = lmda[0] + lmda[1] * x[i] ** 2
        var_term = np.exp(-lmda[0] - lmda[1] * x[i] ** 2)
        l2_term = (y[i] - b[0] - b[1] * x[i]) ** 2
        l += log_var_term + var_term * l2_term
    return l


def b_update_gaussian(b, lmda, x, y, learning_rate):
    b_0_grad, b_1_grad = 0, 0
    n_ = len(x)
    for i in range(n_):
        var_term = np.exp(-lmda[0] - lmda[1] * x[i] ** 2)
        resid_term = y[i] - b[0] - b[1] * x[i]
        b_0_grad += var_term * resid_term
        b_1_grad += x[i] * var_term * resid_term
    b_0_grad *= -2 * n_ ** -1
    b_1_grad *= -2 * n_ ** -1
    b_0 = b[0] - learning_rate * b_0_grad
    b_1 = b[1] - learning_rate * b_1_grad
    return np.array([b_0, b_1])


def lmda_update_gaussian(b, lmda, x, y, learning_rate):
    lmda_0_grad, lmda_1_grad = 0, 0
    n_ = len(x)
    for i in range(n_):
        var_term = np.exp(-lmda[0] - lmda[1] * x[i] ** 2)
        resid_term = (y[i] - b[0] - b[1] * x[i]) ** 2
        lmda_0_grad += var_term * resid_term
        lmda_1_grad += x[i] ** 2 * (1 - var_term * resid_term)
    lmda_0_grad = 1 - n_ ** -1 * lmda_0_grad
    lmda_1_grad *= n_ ** -1
    lmda_0 = lmda[0] - learning_rate * lmda_0_grad
    lmda_1 = lmda[1] - learning_rate * lmda_1_grad
    return np.array([lmda_0, lmda_1])


def neg_log_likelihood_laplacian(b, lmda, x, y):
    l = 0
    n_ = len(x)
    for i in range(n_):
        log_var_term = 0.5 * (lmda[0] + lmda[1] * x[i] ** 2)
        var_term = np.exp(-0.5 * (lmda[0] + lmda[1] * x[i] ** 2))
        l1_term = np.abs(y[i] - b[0] - b[1] * x[i])
        l += log_var_term + var_term * l1_term
    return l


def b_update_laplacian(b, lmda, x, y, learning_rate):
    b_0_grad, b_1_grad = 0, 0
    n_ = len(x)
    for i in range(n_):
        var_term = np.exp(-0.5 * (lmda[0] + lmda[1] * x[i] ** 2))
        sign_term = np.sign(y[i] - b[0] - b[1] * x[i])
        b_0_grad += sign_term * var_term
        b_1_grad += sign_term * x[i] * var_term
    b_0_grad *= -n_ ** -1
    b_1_grad *= -n_ ** -1
    b_0 = b[0] - learning_rate * b_0_grad
    b_1 = b[1] - learning_rate * b_1_grad
    return np.array([b_0, b_1])


def lmda_update_laplacian(b, lmda, x, y, learning_rate):
    lmda_0_grad, lmda_1_grad = 0, 0
    n_ = len(x)
    for i in range(n_):
        var_term = np.exp(-0.5 * (lmda[0] + lmda[1] * x[i] ** 2))
        l1_term = np.abs(y[i] - b[0] - b[1] * x[i])
        lmda_0_grad += var_term * l1_term
        lmda_1_grad += x[i] ** 2 * (1 - var_term * l1_term)
    lmda_0_grad = 0.5 - (2 * n_) ** -1 * lmda_0_grad
    lmda_1_grad *= (2 * n_) ** -1
    lmda_0 = lmda[0] - learning_rate * lmda_0_grad
    lmda_1 = lmda[1] - learning_rate * lmda_1_grad
    return np.array([lmda_0, lmda_1])


def train_gaussian(num_steps, learning_rate):

    print('Finding MLE estimate for Gaussian prior ...')

    for step in range(num_steps):

        # Compute loss
        l_step = 0
        for dataset in files:

            # Extract current parameters from step - 1
            b, lmda = data[dataset]['b_gauss'][-1], data[dataset]['lmda_gauss'][-1]
            x, y = data[dataset]['x'], data[dataset]['y']

            # Parameter updates
            data[dataset]['b_gauss'].append(b_update_gaussian(b, lmda, x, y, learning_rate))
            data[dataset]['lmda_gauss'].append(lmda_update_gaussian(b, lmda, x, y, learning_rate))

            # Compute loss
            if step % print_interval == 0:
                l = neg_log_likelihood_gaussian(b, lmda, x, y)
                l_step += l
                data[dataset]['loss_gauss'].append(l_step)

        # Save loss
        if step % print_interval == 0:
            loss_gauss.append(l_step / float(n))
            print('Loss at step %s: %.4f' % (step, loss_gauss[-1]))


def train_laplacian(num_steps, learning_rate):

    print('Finding MLE estimate for Laplacian prior ...')

    for step in range(num_steps):

        # Compute loss
        l_step = 0
        for dataset in files:

            # Extract model parameters from step - 1
            b, lmda = data[dataset]['b_lap'][-1], data[dataset]['lmda_lap'][-1]
            x, y = data[dataset]['x'], data[dataset]['y']

            # Parameter updates
            data[dataset]['b_lap'].append(b_update_laplacian(b, lmda, x, y, learning_rate))
            data[dataset]['lmda_lap'].append(lmda_update_laplacian(b, lmda, x, y, learning_rate))

            # Compute loss
            if step % print_interval == 0:
                l = neg_log_likelihood_laplacian(b, lmda, x, y)
                l_step += l
                data[dataset]['loss_lap'].append(l_step)

        # Save loss
        if step % print_interval == 0:
            loss_lap.append(l_step / float(n))
            print('Loss at step %s: %.4f' % (step, loss_lap[-1]))


'''############## Misc ##############'''


def load_model():
    print('Loading data from binary ...')
    pickle_file = 'model.pickle'
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data, data.keys()


def save_model():
    savefile = 'model.pickle'
    print('Serializing dataset into %s ...' % savefile)
    try:
        f = open(savefile, 'wb')
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except Exception as exception:
        print('Unable to save data to' + savefile + ': %d' % exception)
        raise
    print("Process Complete.")


def write_csv():
    columns = ['file', 'a', 'b']
    a, b = list(), list()
    df = pd.DataFrame(columns=columns)
    for dataset in files:
        a.append(data[dataset]['b_lap'][-1][0])
        b.append(data[dataset]['b_lap'][-1][1])
    df['file'] = files
    df['a'] = a
    df['b'] = b
    df.to_csv('estimates.csv', sep=',', index=False)


def print_results():

    for dataset in files:

        print('######################################')
        print(dataset)
        print('OLS --> a: %.4f  b: %.4f' % (data[dataset]['b_ols'][-1][0], data[dataset]['b_ols'][-1][1]))
        print('IRLS --> a: %.4f  b: %.4f' % (data[dataset]['b_irls'][-1][0], data[dataset]['b_irls'][-1][1]))
        print('MLE_G --> a: %.4f  b: %.4f' % (data[dataset]['b_gauss'][-1][0], data[dataset]['b_gauss'][-1][1]))
        print('MLE_L --> a: %.4f  b: %.4f' % (data[dataset]['b_lap'][-1][0], data[dataset]['b_lap'][-1][1]))


def plot():

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    count = 0
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(20, 20)
    x_resids = np.array(())

    for dataset in files:

        row, col = count / 3, count % 3
        x, y = data[dataset]['x'], data[dataset]['y']
        x_resids = np.append(x_resids, x)

        n_ = len(x)

        # Data
        ax[row, col].scatter(x, y, color='c', alpha=0.5)

        # OLS fit
        a_ols = np.full((n_,), data[dataset]['b_ols'][-1][0])
        b_ols = np.full((n_,), data[dataset]['b_ols'][-1][1])
        OLS, = ax[row, col].plot(x, a_ols + x * b_ols, '-k', alpha=0.5, label='OLS')

        # IRLS fit
        a_irls = np.full((n_,), data[dataset]['b_irls'][-1][0])
        b_irls = np.full((n_,), data[dataset]['b_irls'][-1][1])
        IRLS, = ax[row, col].plot(x, a_irls + x * b_irls, '-g', alpha=0.5, label='IRLS')

        # MLE-Gaussian
        a_gauss = np.full((n_,), data[dataset]['b_gauss'][-1][0])
        b_gauss = np.full((n_,), data[dataset]['b_gauss'][-1][1])
        x = np.linspace(min(x), max(x), num=n_)
        MLE_G, = ax[row, col].plot(x, a_gauss + x * b_gauss, '-b', alpha=0.75, label='MLE_G')

        # MLE-Laplacian
        a_lap = np.full((n_,), data[dataset]['b_lap'][-1][0])
        b_lap = np.full((n_,), data[dataset]['b_lap'][-1][1])
        x = np.linspace(min(x), max(x), num=n_)
        MLE_L, = ax[row, col].plot(x, a_lap + x * b_lap, '-r', alpha=0.75, label='MLE_L')

        # Labels
        ax[row, col].legend(handles=[OLS, IRLS, MLE_G, MLE_L], fontsize=9, framealpha=0.5)
        ax[row, col].set_xlabel('x')
        ax[row, col].set_ylabel('y')
        ax[row, col].set_title(dataset, fontsize=12)

        count += 1

    # Residuals from IRLS
    ax[1, 2].scatter(x_resids, np.abs(resids), color='c', alpha=0.5)
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel('abs(Y - BX)')
    ax[1, 2].set_title('Residuals from all datasets', fontsize=12)

    # Plot
    plt.show()


if __name__ == '__main__':

    # Define things here
    n = 350
    print_interval = 5000

    # Load data
    data, files = load_data()

    # # If already saved, load here
    # data, files = load_model()

    # OLS estimates
    run_OLS()

    # IRLS estimates
    resids = run_IRLS(steps=10, k=2.)

    # MLE w/ Guassian prior --> GD
    loss_gauss = list()
    train_gaussian(num_steps=100001, learning_rate=9.5e-5)

    # MLE w/ Laplacian prior --> GD
    loss_lap = list()
    train_laplacian(num_steps=100001, learning_rate=2.85e-4)

    # # Create pickle file for reuse
    # save_model()

    # Print out point estimates
    print_results()

    # # Save point estimates to .csv
    # write_csv()

    # Plot data and fits
    plot()
