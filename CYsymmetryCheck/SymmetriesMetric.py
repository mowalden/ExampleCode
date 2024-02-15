import os as os
import pickle as pickle
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

import warnings

tf.get_logger().setLevel('ERROR')

# Suppress UserWarnings and FutureWarnings from TensorFlow
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


"""
    SYMMETRY CHECK

    This code snippet checks whether the trained Ricci-flat metric obeys certain symmetries without being initially constrained, i.e. does the metric learn symmetries of the underlying space?
"""

# Hypersurface definition, needed to check whether a point is on the CY
monomials = 5*np.eye(5, dtype=np.int64)
coefficients = np.ones(5)
kmoduli = np.ones(1)
ambient = np.array([4])
pg = PointGenerator(monomials, coefficients, kmoduli, ambient)


data = np.array([
                    dict(np.load(os.path.join('fermat_pg', 'dataset.npz'))), 
                    dict(np.load(os.path.join('fermat_pg_asym', 'dataset.npz')))
                 ])

model1 = tf.keras.models.load_model('fermat_pg/quintic.keras')
model2 = tf.keras.models.load_model('fermat_pg_asym/quintic_asym.keras')
BASIS1 = np.load(os.path.join('fermat_pg', 'basis.pickle'), allow_pickle=True)
BASIS1 = prepare_tf_basis(BASIS1)
BASIS2 = np.load(os.path.join('fermat_pg_asym', 'basis.pickle'), allow_pickle=True)
BASIS2 = prepare_tf_basis(BASIS2)
alpha = [1.,1.,1.,1.,1.]

metric = np.array([ PhiFSModel(model1, BASIS1, alpha=alpha), 
                    PhiFSModel(model2, BASIS2, alpha=alpha)
                    ]) 

examples = np.array([
                        'Quintic with permutation symmetry', 
                        'Quinctic with permutation symmetry but without scaling symmetry'
                        ])

# Number of examples
n = np.size(examples, axis=0)
# Patch
patch = 0

diff = np.zeros(3)
diffGeneral = np.zeros(2)

for i in range(n):
    
    # Select patch
    data[i]['X_train'] = np.delete(data[i]['X_train'], np.where(data[i]['X_train'][:,patch] != 1.), axis=0)

    # Permutation together with selection of points that are all solved for the same coordinate
    points = data[i]['X_train'][:, 0:pg.ncoords] + 1.j*data[i]['X_train'][:, pg.ncoords:]
    solvedFor = metric[i]._find_max_dQ_coords(tf.convert_to_tensor(data[i]['X_train'], dtype=tf.float32))
    idx = np.where(solvedFor == 1)
    points = points[idx[0]]
    pointsPerm = points.copy()
    pointsPerm[:,2:5] = np.roll(pointsPerm[:,2:5], -1, axis=-1)

    # Scale points
    if i ==0:
        scaleFactor = np.array([1., 1., np.exp(2*np.pi*1./5.j), np.exp(2*np.pi*2./5.j), 
                                np.exp(2*np.pi*3./5.j)])
        scaleFactor = np.repeat(np.expand_dims(scaleFactor, axis=0), len(points), axis=0)
        dataScale = points*scaleFactor
        pointsScaled = np.concatenate((np.real(dataScale), np.imag(dataScale)), axis=-1)
        pointsScaled = tf.convert_to_tensor(pointsScaled, dtype=tf.float32)

    # Compute random difference
    pointsFiltered = np.concatenate((np.real(points), np.imag(points)), axis=-1)
    idx = np.random.randint(np.size(pointsFiltered, axis=0), size=400)
    randomPoints = np.array(np.split(pointsFiltered[idx], 2, axis=0))
    norm = tf.norm(metric[i](randomPoints[0]), axis=[-2,-1])
    diffMatrix = tf.norm(metric[i](randomPoints[0])-metric[i](randomPoints[1]), axis=[-2,-1])
    diffGeneral[i] = tf.math.reduce_mean(diffMatrix/norm).numpy()

    # Compute Jacobian of permutation in patch zero: dz'/dz, where z' is the permuted coordinate
    jacobianPerm = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]).T
    jacobianPerm = np.repeat(np.expand_dims(jacobianPerm, axis=0), np.size(pointsPerm, axis=0), axis=0)


    # Convert point back to cymetric format
    pointsPerm = np.concatenate((np.real(pointsPerm), np.imag(pointsPerm)), axis=-1)
    pointsPerm = tf.convert_to_tensor(pointsPerm, dtype=tf.float32)
    points = np.concatenate((np.real(points), np.imag(points)), axis=-1)
    points = tf.convert_to_tensor(points, dtype=tf.float32)



    # Calculate transformed metric at each point
    beforeTrans = metric[i](points)
    afterPerm = jacobianPerm@metric[i](pointsPerm)@np.transpose(jacobianPerm, axes=(0,2,1))            

    diff[2*i] = tf.math.reduce_mean(tf.norm(beforeTrans - afterPerm,  axis=[-2,-1])/tf.norm(beforeTrans)).numpy()
    # Scaling
    if i == 0:
        # Compute jacobian
        jacobianScale = np.array([[np.exp(2*np.pi*1./5.j), 0.+0.j, 0.+0.j], [0.+0.j, np.exp(2*np.pi*2./5.j), 0.+0.j], [0.+0.j, 0.+0.j, np.exp(2*np.pi*3./5.j)]])
        jacobianScale = np.repeat(np.expand_dims(jacobianScale, axis=0), np.size(dataScale, axis=0), axis=0)

        j_elim = tf.expand_dims(tf.ones(len(points), dtype=tf.int64), axis=-1)

        afterScale = jacobianScale@metric[i](pointsScaled, j_elim=j_elim)@np.transpose(np.conjugate(jacobianScale), axes=(0,2,1))

        diff[i+1] = tf.math.reduce_mean(tf.norm(beforeTrans - afterScale, axis=1)/tf.norm(beforeTrans, axis=1)).numpy()


for i in range(n):
    print('--------------------------------------------------------------')
    print('\n' + examples[i] + ': \n')
    print('Number of points: ' + str(np.size(data[i]['X_train'], axis=0)))
    print('General difference: ' + str(diffGeneral[i]))
    print('Error permutation: ' + str(diff[2*i]))
    if i == 0:
        print('Error scaling: ' + str(diff[i+1]) + '\n')
    else:
        print(' ')

print('--------------------------------------------------------------')






