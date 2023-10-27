import numpy
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # Your implementation goes here!
    dataset = np.load(filename)
    dataset = dataset - np.mean(dataset, axis=0)
    return dataset


def get_covariance(dataset):
    # Your implementation goes here!
    # print(np.transpose(dataset).shape)
    # print(len(dataset) - 1)
    return np.dot(np.transpose(dataset), dataset) / (len(dataset) - 1)


def get_eig(S, m):
    # Your implementation goes here!
    eig_values, eig_vectors = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])

    eig_values = np.flip(eig_values)
    eig_vectors = np.flip(eig_vectors, axis=1)

    diag_eig_values = np.zeros((len(eig_values), len(eig_values)))

    for i, value in enumerate(eig_values):
        diag_eig_values[i][i] = value

    return diag_eig_values, eig_vectors


def get_eig_prop(S, prop):
    # Your implementation goes here!
    all_eigen, _ = get_eig(S, len(S))
    # print(all_eigen)
    sum_eigen = np.trace(all_eigen)
    threshold = prop * sum_eigen
    # print(sum_eigen)
    eig_values, eig_vectors = eigh(S, subset_by_value=[threshold, np.inf])

    eig_values = np.flip(eig_values)
    eig_vectors = np.flip(eig_vectors, axis=1)

    diag_eig_values = np.zeros((len(eig_values), len(eig_values)))

    for i, value in enumerate(eig_values):
        diag_eig_values[i][i] = value

    return diag_eig_values, eig_vectors


def project_image(image, U):
    # Your implementation goes here!
    m = len(U[0])
    result = np.zeros((1, len(U)))
    for j in range(m):
        eigen_vector = U[:, j]
        image = image[:len(eigen_vector)]
        # print(np.transpose(eigen_vector).shape)
        # print(image)
        result += np.dot(np.dot(np.transpose(eigen_vector), image[:len(eigen_vector)]), eigen_vector)
    return result


def display_image(orig, proj):
    if not isinstance(orig, np.ndarray) or not isinstance(proj, np.ndarray):
        raise TypeError("orig and proj must be a NumPy array")

    orig_32x32 = orig.reshape((32, 32), order='F')
    proj_32x32 = proj.reshape((32, 32), order='F')

    figure, (ax_orig, ax_proj) = plt.subplots(nrows=1, ncols=2)
    ax_orig.set_title('Original')
    ax_proj.set_title('Projection')

    img_orig = ax_orig.imshow(orig_32x32, aspect='equal')
    img_proj = ax_proj.imshow(proj_32x32, aspect='equal')

    figure.colorbar(img_orig, ax=ax_orig)
    figure.colorbar(img_proj, ax=ax_proj)

    plt.show()


