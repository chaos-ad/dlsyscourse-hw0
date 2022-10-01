import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE

def unpack_part(fmt, data):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, data[:size]), data[size:]

def read_idx_file(filename):
    with gzip.open(filename, mode='rb') as fileobj:
        data = fileobj.read()

        (zero1, zero2, type_id, dims), data = unpack_part('>bbbb', data)
        if zero1 != 0 or zero2 != 0:
            raise Exception("Invalid file format")

        types = {
            int('0x08', base=16): 'B',
            int('0x09', base=16): 'b',
            int('0x0B', base=16): 'h',
            int('0x0C', base=16): 'i',
            int('0x0D', base=16): 'f',
            int('0x0E', base=16): 'd'
        }
        type_code = types[type_id]

        dim_sizes, data = unpack_part('>' + ('i' * dims), data)
        num_examples = dim_sizes[0]
        input_dim = int(np.prod(dim_sizes[1:]))

        X, data = unpack_part('>' + (type_code * (num_examples * input_dim)), data)
        if data:
            raise Exception("invalid file format")

        new_shape = (num_examples, input_dim) if input_dim > 1 else num_examples
        return np.array(X).reshape(new_shape, order='C')

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    images = read_idx_file(image_filename).astype('float32')
    images = (images - images.min()) / (images.max() - images.min())
    labels = read_idx_file(label_filename).astype('uint8')
    return images, labels
    ### END YOUR CODE

def one_hot(indexes, dims=None):
    dims = dims or indexes.max()+1
    return np.eye(dims)[indexes]

def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # z2 = np.log(np.exp(Z).sum(axis=1))
    # y2 = (Z * one_hot(y)).sum(axis=1)
    # return (z2 - y2).mean()
    return (np.log(np.exp(Z).sum(axis=1)) - Z[np.arange(Z.shape[0]), y]).mean()
    ### END YOUR CODE

def normalize_rows(values):
  return values / values.sum(axis=1)[:, np.newaxis]

def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # X.shape = (num_examples, input_dim)
    # theta.shape = (input_dim, num_classes)
    # print(f"=== START ===")
    num_classes = y.max() + 1
    for i in range(0, X.shape[0], batch):
        # print(f"processing minibatch [{i}, {i+batch})...")
        minibatch_X = X[i:i+batch]
        minibatch_y = y[i:i+batch]
        # print(f"minibatch_X.shape = {minibatch_X.shape}\n\t{minibatch_X}")
        # print(f"minibatch_y.shape = {minibatch_y.shape}\n\t{minibatch_y}")
        # print(f"theta.shape = {theta.shape}\n\t{theta}")
        logodds = np.matmul(minibatch_X, theta) # shape = (num_examples x num_classes)
        # print(f"logodds.shape = {logodds.shape}\n\t{logodds}")
        Z_exp = np.exp(logodds)
        # print(f"Z_exp.shape = {Z_exp.shape}\n\t{Z_exp}")
        Z_norm = normalize_rows(Z_exp) # shape = (num_examples x num_classes)
        # print(f"Z_norm.shape = {Z_norm.shape}\n\t{Z_norm}")
        Y_onehot = one_hot(minibatch_y, dims=num_classes)
        # print(f"Y_onehot.shape = {Y_onehot.shape}\n\t{Y_onehot}")
        Z_sub = (Z_norm - Y_onehot)  # shape = (num_examples x num_classes)
        # print(f"Z_sub.shape = {Z_sub.shape}\n\t{Z_sub}")
        minibatch_X_T = minibatch_X.T
        # print(f"minibatch_X_T.shape = {minibatch_X_T.shape}\n\t{minibatch_X_T}")
        theta_grad = np.matmul(minibatch_X_T, Z_sub) # (input_dim x num_classes)
        # print(f"theta_grad.shape = {theta_grad.shape}\n\t{theta_grad}")
        theta -= (lr / batch) * theta_grad
        # print(f"theta.shape = {theta.shape}\n\t{theta}")

    # print(f"=== FINISH === final teta.shape = {theta.shape}\n\t{theta}")
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_classes = y.max() + 1
    for i in range(0, X.shape[0], batch):
        # print(f"processing minibatch [{i}, {i+batch})...")
        minibatch_X = X[i:i+batch]
        minibatch_y = y[i:i+batch]
        Z1 = np.maximum(0, np.matmul(minibatch_X,W1))
        G2 = normalize_rows(np.exp(np.matmul(Z1,W2))) - one_hot(minibatch_y, dims=num_classes)
        G1 = np.matmul(G2,W2.T) * (Z1 > 0).astype('int')
        W1_grad = np.matmul(minibatch_X.T, G1)
        W2_grad = np.matmul(Z1.T, G2)
        W1 -= (lr / batch) * W1_grad
        W2 -= (lr / batch) * W2_grad
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
