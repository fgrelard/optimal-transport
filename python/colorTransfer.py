import numpy as np
import cv2
import matplotlib.pyplot as plt


def orth_projection(a,b):
    return np.dot(a,b)

PATH="../"

# Load an color image
source = cv2.imread(PATH+'pexelA-0_small.png',cv2.IMREAD_COLOR)
print("Source image " + str(source.shape))

# Load an color image
target = cv2.imread(PATH+'pexelB-0_small.png',cv2.IMREAD_COLOR)
print("Target image " + str(target.shape))

# cv2.imshow('source',source)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('target',target)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##Swapping the channels
output = source.copy()
rows,cols,nbchannels = source.shape


colors_source = [source[i,j].astype(float)/255 for i in range(rows) for j in range(cols)]
colors_target = [target[i,j].astype(float)/255 for i in range(rows) for j in range(cols)]

nb_iterations = 30
nb_batches = 8

energies = []


indices_source = [i for i in range(len(colors_source))]
indices_target = [i for i in range(len(colors_target))]

for n in range(nb_iterations):
    print("Iteration", n)
    current_energy = 0
    vector_batch = [np.array([0., 0., 0.]) for i in range(len(colors_source))]
    for b in range(nb_batches):
        random_direction = [np.random.normal(0, 1.0) for i in range(3)]
        random_direction /= np.linalg.norm(random_direction)
        projections_source, projections_target = [], []
        for i in range(len(indices_source)):
            s = colors_source[i]
            t = colors_target[i]
            projections_source.append(orth_projection(s, random_direction))
            projections_target.append(orth_projection(t, random_direction))
        indices_source = sorted(indices_source, key=lambda x: projections_source[x])
        indices_target = sorted(indices_target, key=lambda x: projections_target[x])
        for i in range(len(indices_source)):
            ind_s = indices_source[i]
            ind_t = indices_target[i]
            if i ==0:
                print(projections_source[ind_s], projections_target[ind_t])
            s_i = projections_source[ind_s] * random_direction
            t_i = projections_target[ind_t] * random_direction
            # current_energy += np.sum((t_i-s_i)**2)
            vector_batch[ind_s] += t_i - s_i
    # energies.append(current_energy)
    for i in range(len(colors_source)):
        new_value = colors_source[i] + (vector_batch[i] / nb_batches)
        new_value = [max(0, min(elem, 1)) for elem in new_value]
        colors_source[i] = new_value


# plt.plot([i for i in range(nb_iterations)], energies)
# plt.show()


for i in range(rows):
  for j in range(cols):
     index = i * cols + j
     new_color = (np.array(colors_source[index]) * 255).astype(np.uint8)
     if i==0 and j ==0:
         print(new_color)
     output[i,j]= new_color

print("Saving the output")
cv2.imwrite('output.png',output)
