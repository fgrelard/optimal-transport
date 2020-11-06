import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cv2

# minimum value of the gaussian
EPSILON = 1E-250

# blurring parameter
gamma = 40

def one_direction_gaussian(image, axis, sigma):
    t = np.linspace(0, image.shape[axis], image.shape[axis])
    [Y, X] = np.meshgrid(t, t)
    return np.maximum(EPSILON, np.exp(-(X - Y) ** 2 / sigma))

def separable_gaussian(image, sigma):
    xi1 = one_direction_gaussian(image, 0, sigma)
    xi2 = one_direction_gaussian(image, 1, sigma)
    return np.dot(np.dot(xi1, image), xi2)


def animate_full(images):
    fig, ax = plt.subplots(figsize=(5,5))
    fig.subplots_adjust(0,0,1,1)
    ax.set_axis_off()
    def update(frame):
        ax.clear()
        ax.set_axis_off()
        ax.imshow(images[frame], cmap="gray")

    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=500, repeat=True)
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("wasserstein_barycenters_bw_m.mp4", writer=writer)

def sinkhorn(f,g,C):
    xi = np.exp(-C/gamma)
    v = np.ones(f.shape)
    for i in range(nb_iterations):
        u = f / (xi * v)
        v = g / (xi * u)
    M = diag(u)*xi*diag(v)
    print(M)
    return M

def regularized_barycenters(images, weights, original_images=None, nb_iterations=100):
    sigma = np.sqrt(gamma/2)
    N = len(images)
    s = images[0].shape
    v = np.ones(shape=(N, s[0], s[1]))
    u = np.zeros(shape=(N, s[0], s[1]))
    for i in range(nb_iterations):
        for j in range(N):
            u[j] = images[j] / separable_gaussian(v[j],gamma)
        g_u = []
        for j in range(N):
            g_u.append(separable_gaussian(u[j],gamma))
        b = np.prod([np.power(g_u[j], weights[j]) for j in range(N)], axis=0)
        # b = np.sum([weights[j] * np.nan_to_num(np.log(g_u[j])) for j in range(N)],axis=0)
        for j in range(N):
            v[j] = b / g_u[j]
    # M = [np.nan_to_num(np.log(1/(u[j]*v[j]/images[j]))) for j in range(N)]
    M = [np.outer(np.diag(u[j]),separable_gaussian(v[j], gamma)) for j in range(N)]
    M = [np.reshape(M[j], (s[0],s[0],s[1])) for j in range(N)]

    rec = [np.sum([separable_gaussian(images[j],gamma)*M[j][i,:,:] for i in range(M[j].shape[0])], axis=0) for j in range(N)]
    rec = [rec[j]/(2*np.sum(rec[j])) for j in range(N)]
    # rec = [np.sum([M[j][i,:,:] for i in range(M[j].shape[0])], axis=0) for j in range(N)]
    # rec = [u[j]*v[j] for j in range(N)]

    fig, ax = plt.subplots(4,3)
    for i in range(N):
        ax[0,i].imshow(images[i])
        # ax[1,i].imshow(M[i])
        ax[2,i].imshow(rec[i])
        print(np.sum(rec[i]))
    # rec = np.sum(rec, axis=0)
    # ax[2,0].imshow(rec)
    # ax[2,1].imshow(b)
    # ax[2,2].imshow(b-rec)
    # ax[3, 0].imshow(separable_gaussian(np.prod([np.power(M[j],weights[j]) for j in range(N)], axis=0), gamma))
    lc = separable_gaussian(images[0],gamma)+np.sum([rec[i]*weights[i] for i in range(N)], axis=0)
    ax[3, 0].imshow(lc)
    ax[3, 1].imshow(b)
    plt.show()
    if original_images is None:
        im_rec = np.sum([weights[j]*M[j] for j in range(N)], axis=0)
    else:
        im_rec = np.sum([original_images[j].T*(weights[j]*M[j]).T for j in range(N)], axis=0)
        min_values = np.sum([weights[j]*np.min(original_images[j]) for j in range(N)])
        max_values = np.sum([weights[j]*np.max(original_images[j]) for j in range(N)])
        im_rec = cv2.normalize(im_rec.T, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        im_rec[:,:, -1] = 1.0

    # W = gamma*np.sum([images[j]*np.nan_to_num(np.log(u[j]))-images[j] + b*np.nan_to_num(np.log(v[j]))-b for j in range(N)])
    # print(W)
    return im_rec

def open_image(filename):
    img = mpimg.imread(filename)
    img = img[:,:,0].copy().astype(np.float)
    img = img.copy().astype(np.float)
    # img = 255.0 - img
    img /= np.sum(img)
    return img

f = np.array([0,5,1,2])
g = np.array([1,3,3,1])
C = np.array([ [i+j for i in range(len(f))] for j in range(len(f))])
print(C)
# loads a grayscale image

# lotus_1 = open_image('lotus/1.bmp')
lotus_2 = open_image('evol1.bmp')
# lotus_3 = open_image('lotus/3.bmp')
lotus_4 = open_image('evol2.bmp')
# lotus_5 = open_image('lotus/5.bmp')
lotus_6 = open_image('evol3.bmp')

path = 'lotus/color/'
lotus_1_rgb = mpimg.imread(path+"1.png")
lotus_2_rgb = mpimg.imread(path+"2.png")
lotus_3_rgb = mpimg.imread(path+"3.png")
lotus_4_rgb = mpimg.imread(path+"4.png")
lotus_5_rgb = mpimg.imread(path+"5.png")
lotus_6_rgb = mpimg.imread(path+"6.png")

# plt.imshow(lotus_1_rgb)

# fig, ax = plt.subplots(1,2)
# ax[0].imshow(horse)
# ax[1].imshow(man)
# plt.show()

nb_shapes = 10
images = []
h = [lotus_2, lotus_4, lotus_6]
o = [lotus_2_rgb, lotus_4_rgb, lotus_6_rgb]
den = nb_shapes / (len(h) - 1.0)
for i in range(nb_shapes):
    print("Iteration", i)
    if i % den == 0:
        images += [o[int(i/den)] for j in range(3)]

    f_i = min(i * 1.0 / den, 1)
    f_j = max(0, (i - den)/den)
    weights = [1 - f_i, f_i-f_j, f_j]
    # weights = [1, 0, 0]
    barycenter = regularized_barycenters(h, weights, nb_iterations=20)
    images.append(barycenter)


images += [o[-1] for i in range(5)]
print(len(images))

animate_full(images)
