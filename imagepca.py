import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from helperfunctions import *

np.random.seed(0) # set random seed for reproducibility

n = len(os.listdir('data'))  # number of files in the data directory
img = plt.imread(f'data/{os.listdir("data")[0]}')  # load the first file

nrow = 64 # number of rows in each image
ncol = 64 # number of columns in each image
npix = nrow * ncol # number of pixels in each image
data = np.zeros((n, npix))  # initialize data array

if os.path.exists(f'data_{nrow}_{ncol}.npz'):
    with np.load(f'data_{nrow}_{ncol}.npz') as f:
        data = f['data']
        n = len(data)
else:
    i = 0
    skips = np.zeros(n, dtype=bool)
    for file in os.listdir('data'):
        try:
            img = plt.imread(f'data/{file}')
        except:
            print(f"bad file: {file}, skipping")
            skips[i] = 1
            continue
        imgbw = img.mean(axis=2)  # convert to grayscale
        try:
            imgbw = extract_square(imgbw, nrow, ncol)  # extract square
        except AssertionError:
            skips[i] = 1
            continue
        data[i,:] = imgbw.flatten()  # store flattened image in data array
        i += 1
    n = n - sum(skips)
    data = data[~skips, :]
    np.savez(f'data_{nrow}_{ncol}.npz', data=data)

N = n - 1 # for the usual unbiased estimator (ddof=1)

## center data
means = np.mean(data, axis=0) # mean of each pixel
centered = data - means # the data matrix with the mean of each pixel subtracted; "broadcasting" is done automatically as it's a mean over columns not rows


## compute PCA

# the expnsive step--the SVD--is saved so we don't have to recompute it every time
if os.path.exists(f'pca_{nrow}_{ncol}.npz'):
    with np.load(f'pca_{nrow}_{ncol}.npz') as f:
        U = f['U']
        s = f['s']
        VT = f['VT']
else:
    U, s, VT = np.linalg.svd(centered)
    np.savez(f'pca_{nrow}_{ncol}.npz', U=U, s=s, VT=VT)

C = np.dot(centered.T, centered) / N # empirical covariance matrix
V = VT.T # principal components (covariance matrix eigenvectors) in columns of V
l = s**2 / N # eigenvalues

plt.figure()
plt.plot(1+np.arange(len(l)), 10*np.log10(l/sum(l)), 'o-')
plt.title("Explained variance ratio")
plt.xlabel("Principal component number")
plt.ylabel(r"$10\cdot \log_{10}$ explained variance ratio")
plt.savefig(r'explained_variance_ratio_{nrow}_{ncol}.png')
plt.close()


## plot example marginal distribution, for 10 random pixels
for i in np.random.choice(data.shape[1], 10): # choose 10 random pixels
    sigma = np.sqrt(C[i,i]) # modeled marginal standard deviation
    m = means[i] # modeled marginal mean
    xs = np.linspace(m-4*sigma, m+4*sigma, 1000)

    plt.figure()
    plt.plot(xs, stats.norm.pdf(xs, m, sigma), color='k', label=f'marginal (from empirical-covariance-based model)')
    plt.hist(data[:,i], bins=20, alpha=0.5, color='g', label=f'empirical marginal', density=True, histtype='step')
    plt.legend()
    plt.title(f'Pixel {i}')
    plt.savefig(f'hist_marginal_pixel_{i}.png')
    plt.close()

ncomp = 5  # number of principal components to use for reconstruction

## plot principal components and mean
plt.figure()
plt.imshow(means.reshape((nrow, ncol)), cmap='gray')
plt.title("Mean image")
plt.colorbar()
plt.axis('off')
plt.savefig('mean_image.png')

for c in range(ncomp):
    plt.figure()
    plt.imshow(V[:,c].reshape((nrow, ncol)), cmap='gray')
    plt.title(f"principal component {c+1}")
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'principal_component_{c+1}.png')
    plt.close()

## plot example reconstructed images, using PCA model, for 10 random images
for i in np.random.choice(n, 10):  # choose 10 random images
    img_i = data[i, :].reshape((1,npix))  # get the i-th image
    projected_i = (img_i - means) @ V[:, :ncomp]  # project the centered image onto the first nc principal components
    reconstructed_i = V[:,:ncomp] @ projected_i.flatten() + means  # reconstruct the image from the projected data

    plt.figure()
    plt.imshow(reconstructed_i.reshape((nrow, ncol)), cmap='gray')
    plt.axis('off')
    plt.title(f"Reconstructed image {i}, using {ncomp} principal components")
    plt.savefig(f'reconstructed_image_{i}_using_{ncomp}_pcs.png')
    plt.close()

    plt.figure()
    plt.imshow(np.abs(img_i - reconstructed_i).reshape((nrow, ncol)), cmap='gray')
    plt.axis('off')
    plt.title(f"Reconstruction error, image {i}, using {ncomp} principal components")
    plt.colorbar()
    plt.savefig(f'reconstructed_error_image_{i}_using_{ncomp}_pcs.png')
    plt.close()

    ## IF WE HAVE A SMALL PATCH SIZE RELATIVE TO A DATA SET OF FAIRLY SMOOTH IMAGES, WE CAN SEE THAT THE PCA COMPONENTS ARE SIMILAR TO THE LAPLACE-BELTRAMI EIGENFUNCTIONS


img_graph = Image(nrow, ncol)
L, A, D = img_graph.get_laplacian_adjacency_degree_matrix()
eigval, eigvec = np.linalg.eigh(L)
for i in range(ncomp):
    plt.figure()
    plt.imshow(eigvec[:, i].reshape((nrow, ncol)), cmap='gray')
    plt.title(f"Laplacian eigenfunction {i+1}")
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'laplacian_eigenfunction_{i+1}.png')
    plt.close()


## plot principal component smoothness and Laplacian eigenvector smoothness
plt.figure()
for i in range(npix):
    if i == 0:
        plt.plot(i+1, np.dot(V[:,i], np.dot(L, V[:,i])), 'ro', label=f'Covariance eigenfunction smoothness')
        plt.plot(i+1, np.dot(eigvec[:,i], np.dot(L, eigvec[:,i])), 'bx', label='Laplacian eigenfunction smoothness')
    else:
        plt.plot(i+1, np.dot(V[:,i], np.dot(L, V[:,i])), 'ro')
        plt.plot(i+1, np.dot(eigvec[:,i], np.dot(L, eigvec[:,i])), 'bx')
plt.legend()
plt.savefig(f"smoothness_comparison_{nrow}_{ncol}.png")


