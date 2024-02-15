import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

import pandas as pd
import numpy as np
import tensorflow as tf

from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.callbacks import RicciCallback, SigmaCallback, VolkCallback, KaehlerCallback, TransitionCallback
from cymetric.models.tfmodels import MultFSModel, PhiFSModel
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, VolkLoss, RicciLoss, TotalLoss


"""
    SYMMETRY CHECK

    The Fubini-Study metric obeys certain symmetries by construction. This is code snippet serves as a test, since we know that the FS metric pulled back to the CY space must be invariant, hence the error must be zero.
"""

# Define hypersurface
monomials = 5*np.eye(5, dtype=np.int64)
coefficients = np.ones(5)
kmoduli = np.ones(1)
ambient = np.array([4])
pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
points = pg.generate_points(1000)


# Select patch (must be zero for Jacobian to make sense)
patch = 0
points = np.delete(points, np.where(np.real(points[:,patch]) != 1.), axis=0)
solvedFor = pg._find_max_dQ_coords(points) 
idx = np.where(solvedFor == 1)
points = points[idx]


# Perform patch preserving (i.e. acyclic) permutation
pointsPermuted = points.copy()
pointsPermuted[:,2:5] = np.roll(pointsPermuted[:,2:5], -1, axis=-1)

# Compute Jacobian of permutation in patch zero: dz'/dz, where z' is the permuted coordinate
jacobianPerm = np.array([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.],[0.,0.,0.,0.,1.],[0.,0.,1.,0.,0.]]).T
jacobianPerm = np.repeat(np.expand_dims(jacobianPerm, axis=0), np.size(points, axis=0), axis=0)


# Metric before permutation in ambient space
FSinitial = pg(points)


# Metric after permutation in ambient space
FSprime = pg(pointsPermuted)

# J g J^T
FSPerm = jacobianPerm@FSprime@np.transpose(jacobianPerm, axes=(0,2,1))

# Averaged difference in ambient space
diffAmb = tf.math.reduce_mean(tf.norm(FSinitial - FSPerm, axis=[-2,-1])/tf.norm(FSinitial, axis=[-2,-1])).numpy()

# Metric before permutation pulled back on CY
jacobianpb = pg.pullbacks(points)
FSpb = jacobianpb@FSinitial@np.transpose(jacobianpb, axes=(0,2,1))

# Metric after permutation pulled back on CY
jacobianPermpb = pg.pullbacks(pointsPermuted)
FSprimepb = jacobianPermpb@FSprime@np.transpose(jacobianPermpb, axes=(0,2,1))


# Calculate transformed metric
jacobian = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
jacobian = np.repeat(np.expand_dims(jacobian, axis=0), np.size(points, axis=0), axis=0)
FSprimepbPerm = jacobian@FSprimepb@np.transpose(jacobian, axes=(0,2,1))

# Some debugging snippets 
# commonFactorDEBUG = np.power(-1.-points[2,2]**5-points[2,3]**5-points[2,4]**5, -4./5.)
# jacobianDEBUG = np.array([[0.+0.j,1.+0.j,0.+0.j],[0.+0.j,0.+0.j,1.+0.j],[-points[2,2]**4*commonFactorDEBUG,-points[2,3]**4*commonFactorDEBUG,-points[2,4]**4*commonFactorDEBUG]])

# Average difference after pullback
diffpb = tf.math.reduce_mean(tf.norm(FSpb - FSprimepbPerm, axis=[-2,-1])/tf.norm(FSpb, axis=[-2,-1])).numpy()


# # DEBUG
# print('-------------------------------------------------------------------')
# print('Point:')
# point = 5
# print(points[point])
# print('-------------------------------------------------------------------')
# print('Solved for:')
# print(solvedFor[point])
# print('-------------------------------------------------------------------')
# print('Remaining coordinates:')
# print(affinePts[point])
# print('-------------------------------------------------------------------')
# print('Common factor:')
# print(commonFactor[point])
# print('-------------------------------------------------------------------')
# print('Jacobian:')
# print(jacobian[point])
# print('-------------------------------------------------------------------')
# print('Error:')
# print(tf.math.reduce_mean(tf.norm(FSpb[point] - FSprimepbPerm[point])/tf.norm(FSpb[point])).numpy())
# print('-------------------------------------------------------------------')

print('-------------------------------------------------------------------')
print('Average difference in ambient space:')
print(diffAmb)
print('-------------------------------------------------------------------')
print('Average difference on CY-threefold')
print(diffpb)
print('-------------------------------------------------------------------')


