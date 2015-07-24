from stupid_tricks import *
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
import random
from functools import reduce, partial
from operator import add

class KMeans:
  """good old class based solution"""
  def __init__(self, k):
    self.k = k
    self.means = [None for _ in range(k)]

  def fit(self, points, num_iters=10):
    assignments = [None for _ in points]
    self.means = random.sample(list(points), self.k)
    for _ in range(num_iters):
      for i, point in enumerate(points):
        d_min = float('inf')
        for j, m in enumerate(self.means):
          d = sum((m_i - p_i)**2 for m_i, p_i in zip(m, point))
          if d < d_min:
            assignments[i] = j
            d_min = d
      for j in range(self.k):
        cluster = [p for p, c in zip(points, assignments) if c == j]
        self.means[j] = list(map(lambda x: x / len(cluster), reduce(partial(map, add), cluster)))

  def predict(self, point):
    d_min = float('inf')
    for j, m in enumerate(self.means):
      d = sum((m_i - p_i)**2 for m_i, p_i in zip(m, point))
      if d < d_min:
        prediction = j
        d_min = d
    return prediction

def run_kmeans(seed=1):
  random.seed(seed)
  points = np.random.random((100,2))
  
  model = KMeans(5)
  model.fit(points, num_iters=100)
  assignments = [model.predict(point) for point in points]

  for x, y in model.means:
    plt.plot(x, y, marker='*', markersize=10, color='black')

  for j, color in zip(range(5),
                      ['r', 'g', 'b', 'm', 'c']):
    cluster = [p
               for p, c in zip(points, assignments)
               if j == c]
    xs, ys = zip(*cluster)
    plt.scatter(xs, ys, color=color)

  plt.show()

# functional version
# ------------------

def k_meanses(points, k):
  initial_means = random.sample(points, k)
  return iterate(partial(new_means, points),
                 initial_means)

def no_repeat(prev, curr):
  if prev == curr: raise StopIteration
  else: return curr

def until_convergence(it):
  return accumulate(it, no_repeat)

def new_means(points, old_means):
  k = len(old_means)
  assignments = [closest_index(point, old_means)
                 for point in points]
  clusters = [[point
               for point, c in zip(points, assignments)
               if c == j] for j in range(k)]
  return [cluster_mean(cluster) for cluster in clusters]

def closest_index(point, means):
  return min(enumerate(means),
             key=lambda pair: squared_distance(point, pair[1]))[0]

def squared_distance(p, q):
  return sum((p_i - q_i)**2 for p_i, q_i in zip(p, q))

def cluster_mean(points):
  num_points = len(points)
  dim = len(points[0]) if points else 0
  sum_points = [sum(point[j] for point in points)
                for j in range(dim)]
  return [s / num_points for s in sum_points]


def run_kmeans_functional(seed=0):
  random.seed(seed)
  data = [(random.random(), random.random()) for _ in range(500)]
  meanses = [mean for mean in until_convergence(k_meanses(data, 5))]

  x, y = zip(*data)
  plt.scatter(x, y, color='black')

  colors = ['r', 'g', 'b', 'c', 'm']
  for i, means in enumerate(meanses):
    for m, color in zip(means, colors):
      plt.plot(*m, color=color,
               marker='*',
               markersize=3*i)

  plt.show()


def run_kmeans_animation(seed=0, k=5):
  random.seed(seed)
  data = [(random.random(), random.random()) for _ in range(500)]
  meanses = [mean for mean in until_convergence(k_meanses(data, k))]

  # colors = random.sample(list(matplotlib.colors.cnames), k)
  colors = ['r', 'g', 'b', 'c', 'm']

  def animation_frame(nframe):
    means = meanses[nframe]
    plt.cla()
    assignments = [closest_index(point, means)
                   for point in data]
    clusters = [[point
                 for point, c in zip(data, assignments)
                 if c == j] for j in range(k)]

    for cluster, color, mean in zip(clusters, colors, means):
      x, y = zip(*cluster)
      plt.scatter(x, y, color=color)
      plt.plot(*mean, color=color, marker='*', markersize=20)

  fig = plt.figure(figsize=(5,4))
  anim = animation.FuncAnimation(fig, animation_frame, frames=len(meanses))
  anim.save('kmeans_cluster.gif', writer='imagemagick', fps=4)

def run_kmeans_animation2(seed=0, k=5):
  random.seed(seed)
  data = [(random.choice([0,1,2,4,5]) + random.random(),
           random.normalvariate(0, 1)) for _ in range(500)]
  meanses = [mean for mean in until_convergence(k_meanses(data, k))]

  # colors = random.sample(list(matplotlib.colors.cnames), k)
  colors = ['r', 'g', 'b', 'c', 'm']

  def animation_frame(nframe):
    means = meanses[nframe]
    plt.cla()
    assignments = [closest_index(point, means)
                   for point in data]
    clusters = [[point
                 for point, c in zip(data, assignments)
                 if c == j] for j in range(k)]

    for cluster, color, mean in zip(clusters, colors, means):
      x, y = zip(*cluster)
      plt.scatter(x, y, color=color)
      plt.plot(*mean, color=color, marker='*', markersize=20)

  fig = plt.figure(figsize=(5,4))
  anim = animation.FuncAnimation(fig, animation_frame, frames=len(meanses))
  anim.save('kmeans2.gif', writer='imagemagick', fps=4)
