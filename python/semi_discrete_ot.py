#!/usr/bin/env python
# coding: utf-8

from scipy.spatial import Delaunay
import shapely.geometry as geom
from descartes import PolygonPatch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import random
import math

def segment_bisector_intersect(pt0, pt1, site0, site1):
    # intersection between the segment between pt0 and pt1
    # and the bisector seperating the sites site0 and site1
    #
    # the intersection p is defined as pt0 + t(pt1 - pt0) with t in [0,1]
    # the intersection is the point such that p is on the bisector
    # p.(site1 - site0) = 0.5*(site1 +site0).(site1 - site0)

    t = np.dot(site1 + site0 - 2 * pt0, site1 - site0)
    t /= 2 * np.dot(pt1 - pt0, site1 - site0)
    if t >= 0 and t <= 1:
        return pt0 + t * (pt1 - pt0)
    return None

def clip_convex_by_bisector(polygon, site0, site1):
    # Sutherland Hodgman algorithm
    #
    # polygon is the array of vertices of the polygon
    # the idea is to loop on the vertices, keep those
    # on the good side of the bisector and leave those
    # outside. Each portion of outside points is replaced
    # by the pair of intersection points between the sides
    # of the polygon crossing the bisector and the bisector.

    size = len(polygon)

    # the distance to the sites provides the "in or out" information
    d0 = [np.linalg.norm(p - site0) for p in polygon]
    d1 = [np.linalg.norm(p - site1) for p in polygon]

    # storage for the resulting polygon
    result = []
    for i in range(size):
        # previous vertex to determine if the segment crosses the bisector
        previ = (i + size - 1) % size
        if d0[i] < d1[i]:
            #the vertex is inside
            if d0[previ] > d1[previ]:
                #the previous was outside
                # add the intersection
                inter = segment_bisector_intersect(polygon[previ], polygon[i], site0, site1)
                result.append(inter)
            result.append(polygon[i])
        else:
            #the vertex is outside
            if d0[previ] < d1[previ]:
                #the previous was inside
                #add the intersection
                inter = segment_bisector_intersect(polygon[previ], polygon[i], site0, site1)
                result.append(inter)
    return np.array(result)

def clip_by_bisector(shape, site0, site1):
    # shapely provides a polygon / polygon intersection, no required convexity
    # we convert the bisector to a polygon by clipping the shape's bbox
    if shape.is_empty:
        return shape
    bbox = geom.box(*shape.bounds)

    # the shape is 2D but the diagram is potentially of a higher dimension, restricted to xy
    lifted_bbox = np.hstack([
        np.array(bbox.exterior)[:-1],
        np.zeros((4, len(site0) - 2))
    ])

    # bounding box intersection with the sites bisector
    clipped = clip_convex_by_bisector(lifted_bbox, site0, site1)

    # generic shapely clipping
    if len(clipped) > 2:
        return shape.intersection(geom.Polygon(clipped))

    # empty polygon if the bbox was outside the bisector
    return geom.Polygon()

def clip_by_cell(shape, delaunay, index):
    # restricted cell computation
    # iteratively clipping by the bisectors of every Delaunay neighbor
    point = delaunay.points[index]
    nranges, nindices = delaunay.vertex_neighbor_vertices
    neighbors = nindices[nranges[index]:nranges[index+1]]
    result = shape
    for n in neighbors:
        result = clip_by_bisector(result, point, delaunay.points[n])
    return result

def restricted_voronoi(shape, sites):
    # Voronoi diagram restricted to a 2D shape
    # provides an iterator on the restricted cells
    delaunay = Delaunay(sites)
    for i,_ in enumerate(sites):
        yield clip_by_cell(shape, delaunay, i)

def plot_cells(cells, sites = None, radii = None, ax = None):
    # utility to plot a set of polygons.
    if ax is None:
        _,ax= plt.subplots(figsize=(5,5))

    ax.set_axis_off()

    cmap = plt.cm.get_cmap('jet')
    for i,cell in enumerate(cells):
        if not cell.is_empty:
            norm = (np.linalg.norm(np.array([0.5, 0]) - np.array(cell.centroid))-0.1)/0.1
            patch = PolygonPatch(cell, fc=cmap(norm), linewidth=0)
            ax.add_patch(patch)

    if sites is not None:
        colors = [cmap(i%len(cells)) for i,_ in enumerate(sites)]
        ax.scatter(sites[:,0], sites[:,1], s = radii, c = colors, zorder=2, edgecolor='black')

def lloyd_step(shape, sites):
    cells = restricted_voronoi(shape, sites)
    return np.stack([
        # the site is not moved if its restricted cell is empty
        site if cell.is_empty else np.array(cell.centroid)
        for site,cell in zip(sites,cells)
    ])

def restricted_power_cells(domain, sites, weights):
    # ensuring that all weights are negative
    max_weight = np.max(weights)
    lifted_sites = np.hstack([
        sites,
        np.sqrt(max_weight - weights)[:,np.newaxis]
    ])
    return restricted_voronoi(domain, lifted_sites)

def cell_areas(cells):
    return np.array([cell.area for cell in cells])

def gradient_descent_weights(domain, sites, weights, nb_iterations, epsilon=0.5):
    new_weights = weights.copy()
    target_area = domain.area / len(sites)
    target_radius = np.sqrt(target_area/np.pi)
    print(domain.area, target_area)
    sequence_weights = []
    for i in range(nb_iterations):
        cells = list(restricted_power_cells(domain, sites, new_weights))
        areas = cell_areas(cells)
        norm = areas - target_area
        w_d = np.sum(np.abs(norm))
        new_weights = np.array([elem - norm[i]*epsilon
                                for i,elem in enumerate(new_weights)])
        sequence_weights.append(new_weights)
        print(w_d)
        if w_d < 0.001:
            break
    # animate_diagram(domain, sites, sequence_weights)
    return cells, new_weights


def animate_diagram(domain, sites, sequence_weights):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_axis_off()
    def update(frame):
        ax.clear()
        cells = list(restricted_power_cells(domain, sites, sequence_weights[frame]))
        plot_cells(cells,ax = ax)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(sequence_weights), interval=80, repeat=True)
    plt.show()

def animate_diagram_sites(domain, sequence_sites, weights):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_axis_off()
    def update(frame):
        ax.clear()
        cells = list(restricted_power_cells(domain, sequence_sites[frame], weights))
        plot_cells(cells, sequence_sites[frame], ax = ax)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(sequence_sites), interval=1000, repeat=True)
    plt.show()


def animate_full(cells):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    def update(frame):
        ax.clear()
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0, 0.5)
        plot_cells(cells[frame], ax = ax)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(cells), interval=500, repeat=True)
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=4, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("semi_discrete_rainbow.mp4", writer=writer)

def lloyd_like_step(shape, sites, weights):
    cells = restricted_power_cells(shape, sites, weights)
    return np.stack([
        # the site is not moved if its restricted cell is empty
        site if cell.is_empty else np.array(cell.centroid)
        for site,cell in zip(sites,cells)
    ])

def convergence_lloyd_like(shape, sites, weights):
    new_sites = sites.copy()
    sequence_sites = []
    for i in range(1):
        new_sites = lloyd_like_step(shape, new_sites, weights)
        sequence_sites.append(new_sites)
    # animate_diagram_sites(shape, sequence_sites, weights)
    return new_sites


#example
n = 500
R = 0.2
r = 0.1
tx = 0.5
domain = geom.Point(tx,0.).buffer(R).difference(geom.Point(tx,0).buffer(r)).difference(geom.Point(tx,-np.sqrt(2)).buffer(np.sqrt(2)))
angles = [np.random.uniform(0, np.pi) for i in range(n)]
sites = np.array([[np.cos(angles[i])*(r)+tx, np.sin(angles[i])*(r)] for i in range(len(angles))])
print(sites)
# sites = np.random.random_sample((n, 2))

weights = (np.random.random_sample(len(sites)) - 0.5) / n


cells = list(restricted_power_cells(domain, sites, weights))
plot_cells(cells, sites, 1)
plt.show()
new_weights = weights.copy()
new_sites = sites.copy()

evolution=[]
for i in range(100):
    evolution.append(cells)
    cells,  new_weights = gradient_descent_weights(domain, new_sites, new_weights, 100, 0.05)
    new_sites = convergence_lloyd_like(domain, new_sites, new_weights)

animate_full(evolution)

plot_cells(cells, new_sites, 1)
plt.show()
