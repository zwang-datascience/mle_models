import glob
import numpy as np
from sklearn.cross_validation import KFold
from six.moves import cPickle as pickle
from collections import OrderedDict


def load_data():

    print('Loading data ...')

    data = OrderedDict()
    for file_ in glob.glob('*.csv'):

        if file_.split('.')[0] != 'estimates':

            data_ = OrderedDict()

            # Extract data from csv and load into vars x, y
            raw_data = np.genfromtxt(file_, delimiter=',', skip_header=True)
            data_['x'] = raw_data[:, 0].flatten()
            data_['y'] = raw_data[:, 1].flatten()

            # Create nested dicts to store loss and model parameters
            data_['b'] = {0: [np.array([0.1, 0.])], 1: [np.array([0.1, 0.])], 2: [np.array([0.1, 0.])], 3: [np.array([0.1, 0.])], 4: [np.array([0.1, 0.])]}
            data_['lmda'] = {0: [np.array([0.1, 0.])], 1: [np.array([0.1, 0.])], 2: [np.array([0.1, 0.])], 3: [np.array([0.1, 0.])], 4: [np.array([0.1, 0.])]}
            data_['loss_Tr'] = {0: [], 1: [], 2: [], 3: [], 4: []}
            data_['loss_Te'] = {0: [], 1: [], 2: [], 3: [], 4: []}
            data_['step'] = {0: [], 1: [], 2: [], 3: [], 4: []}

            # Save to ordered dict
            data[file_.split('.')[0]] = data_

    files = data.keys()
    print('Data loaded into: %s' % files)

    return data, files


def neg_log_likelihood_laplacian(b, lmda, x, y):
    l = 0
    n_ = len(x)
    for i in range(n_):
        log_var_term = 0.5 * (lmda[0] + lmda[1] * x[i] ** 2)
        var_term = np.exp(-0.5 * (lmda[0] + lmda[1] * x[i] ** 2))
        l1_term = np.abs(y[i] - b[0] - b[1] * x[i])
        l += log_var_term + var_term * l1_term
    return l / float(n_)


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


def run_k_fold_cv(num_steps, learning_rate, k):

    for dataset in files:

        print('Dataset %s' % dataset)

        # Extract data
        x = data[dataset]['x']
        y = data[dataset]['y']

        # Generate training and validation splits
        kf = KFold(len(x), n_folds=k, shuffle=True)

        # Run k-fold cross-cv
        k = 0
        for ind_Tr, ind_Te in kf:

            print('Fold %s' % (k + 1))

            for step in range(num_steps):

                # Extract model parameters from step - 1
                b, lmda = data[dataset]['b'][k][-1], data[dataset]['lmda'][k][-1]
                x_Tr, y_Tr, x_Te, y_Te = x[ind_Tr], y[ind_Tr], x[ind_Te], y[ind_Te]

                # Parameter updates
                data[dataset]['b'][k].append(b_update_laplacian(b, lmda, x_Tr, y_Tr, learning_rate))
                data[dataset]['lmda'][k].append(lmda_update_laplacian(b, lmda, x_Tr, y_Tr, learning_rate))

                # Compute loss
                if step % print_interval == 0 or step in range(0, 10000, 500):
                    l_Tr = neg_log_likelihood_laplacian(b, lmda, x_Tr, y_Tr)
                    l_Te = neg_log_likelihood_laplacian(b, lmda, x_Te, y_Te)
                    data[dataset]['loss_Tr'][k].append(l_Tr)
                    data[dataset]['loss_Te'][k].append(l_Te)
                    data[dataset]['step'][k].append(step)
                    print('Step %s loss: Tr: %.4f, Te: %.4f' % (step, l_Tr, l_Te))

            k += 1

            print('###############################################')


def save_model():
    savefile = 'kfold_cv.pickle'
    print('Serializing dataset into %s ...' % savefile)
    try:
        f = open(savefile, 'wb')
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except Exception as exception:
        print('Unable to save data to' + savefile + ': %d' % exception)
        raise
    print("Process Complete.")


def plot():

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    count = 0
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(20, 20)

    for dataset in files:

        row, col = count / 3, count % 3
        steps = data[dataset]['step'][0]

        # Average losses over k-folds
        n_ = len(steps)
        l_Tr, l_Te = np.ndarray(shape=(1, n_)), np.ndarray(shape=(1, n_))
        for i in range(5):
            l_Tr = np.vstack((l_Tr, data[dataset]['loss_Tr'][i]))
            l_Te = np.vstack((l_Te, data[dataset]['loss_Te'][i]))
        l_Tr, l_Te = np.mean(l_Tr[1:, :], axis=0), np.mean(l_Te[1:, :], axis=0)

        # Plot data
        train, = ax[row, col].plot(steps, l_Tr, color='c', alpha=0.5, label='Train')
        test, = ax[row, col].plot(steps, l_Te, color='m', alpha=0.5, label='Test')

        # Labels & axis
        rng = [[2, 2.5], [3, 4], [3, 5], [2, 3], [2.5, 3.5]]
        ax[row, col].legend(handles=[train, test], fontsize=9, framealpha=0.5)
        ax[row, col].set_xlabel('Number of training steps')
        ax[row, col].set_ylabel('Loss')
        ax[row, col].set_ylim(rng[count])
        ax[row, col].set_title(dataset, fontsize=12)

        count += 1

    # Plot
    plt.show()


if __name__ == '__main__':

    # Define things here
    n = 350
    k = 5
    print_interval = 5000

    # Load data
    data, files = load_data()

    # Run k-fold Cross Validation on MLE-L model
    run_k_fold_cv(num_steps=100001, learning_rate=2.85e-4, k=k)

    # # Save information
    # save_model()

    # Plot train and test loss
    plot()
