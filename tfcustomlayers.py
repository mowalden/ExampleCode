import tensorflow as tf
import numpy as np

"""
    CUSTOM LAYERS

    Various tests on custom group equivariant layers for the cymetric package.
"""



class scaleConv(tf.keras.layers.Layer):

    """
        Custom layer which is equivariant with respect to the Z_5 symmetry of the Fermat quintic. 
        
        PROBLEM: This generates a layer of width 3125 which does not run on a 16 gb ram laptop. This seems to be a general problem with wide layers in tensorflow.
    """

    def __init__(self):

        super(scaleConv, self).__init__()
        self.u = self.add_weight(
        shape=(10,), initializer="random_normal", trainable=True
        )
        self.groupElements = self.generateElements()

    
    def generateElements(self):

        """
        We consider 5th roots of unity of the form exp(2*pi*i*k/5). For the input data given by cymetric (i.e. real and imaginary part split) this results into rotation matrices which are generated below.
        """

        elements = [1./5, 2./5, 3./5, 4./5, 1.,]

        # Generate all possible combinations of 'elements'
        combinations = tf.stack(tf.meshgrid(*[elements]*len(elements)), axis=-1)
        combinations = tf.reshape(combinations, [-1, len(elements)])

        # Use all combinations to create representations in accordance to the cymetric input
        diag = tf.linalg.set_diag( tf.repeat( tf.expand_dims(tf.zeros([5,5], dtype=tf.float32), axis=0), len(combinations), axis=0), tf.math.cos(2.*np.pi*combinations) )
        offDiagPositive = tf.linalg.set_diag( tf.repeat( tf.expand_dims(tf.zeros([5,5], dtype=tf.float32), axis=0), len(combinations), axis=0), tf.math.sin(2.*np.pi*combinations) )
        offDiagNegative = tf.linalg.set_diag( tf.repeat( tf.expand_dims(tf.zeros([5,5], dtype=tf.float32), axis=0), len(combinations), axis=0), -tf.math.sin(2.*np.pi*combinations) )

        return tf.concat( [tf.concat([diag, offDiagNegative], axis=-1), tf.concat([offDiagPositive, diag], axis=-1)], axis=1)

        
    def call(self, inputs):

        # Apply all group elements to a filter of the same shape as the cymetric input and stack them together to form a group equivariant layer
        self.A = tf.einsum('ijk,ik->ij', self.groupElements, tf.repeat(tf.expand_dims(self.u, axis=0), len(self.groupElements), axis=0))
        return tf.nn.gelu(inputs@tf.transpose(self.A))
    
        """
        TEST SNIPPET

        Reducing the amount of rows makes this computable on a 16gb ram laptop. However, this is then not a strictly equivariant layer since most group elements are excluded. 
        """
        # self.A = tf.einsum('ijk,ik->ij', self.groupElements[:100], tf.repeat(tf.expand_dims(self.u, axis=0), 100, axis=0))
        # return tf.nn.gelu(inputs@tf.transpose(self.A))


class simpleScaleConv(tf.keras.layers.Layer):

    """
        Custom layer which is equivariant with respect to the Z_5 symmetry of the Fermat quintic. Simpler version, which commutes with all group elements of the symmetry.
        
    """

    def __init__(self):

        super(simpleScaleConv, self).__init__()
        self.u = self.add_weight(
        shape=(10,), initializer="random_normal", trainable=True
        )

        
    def call(self, inputs):

        diag = tf.linalg.set_diag( tf.zeros([5,5], dtype=tf.float32), self.u[:5] )
        offDiagPositive = tf.linalg.set_diag( tf.zeros([5,5], dtype=tf.float32), self.u[5:] )
        offDiagNegative = tf.linalg.set_diag( tf.zeros([5,5], dtype=tf.float32), -self.u[5:] )
        self.A = tf.concat( [tf.concat([diag, offDiagNegative], axis=-1), tf.concat([offDiagPositive, diag], axis=-1)], axis=0)

        return tf.nn.gelu(inputs@tf.transpose(self.A))
    


    
class permLayer10(tf.keras.layers.Layer):

    """
        Custom layer which is equivariant with respect to the permutation symmetry of the Fermat quintic. This uses the parameter sharing scheme. Hence, the layer is always 10x10.

        PROBLEM:
        Stacking several of these layers does not run on a 16 gb ram laptop. Inserting one of these layers in front of the 'normal' setup used in the tutorial does improve the performance on the transition loss but worsen the performance on the MA loss.
    """

    def __init__(self):
        super(permLayer10, self).__init__()
        self.u = self.add_weight(
            shape=(8,), initializer="random_normal", trainable=True
        ) 


    def call(self, inputs):

        # Impose the appropriate parameter sharing for the full permutation group. Again, the specific form of the input restricts to a slightly different form.
        U = tf.fill([5,5], self.u[0])
        U = tf.linalg.set_diag(U, tf.repeat(tf.expand_dims(self.u[1], 0), 5))
        U1 = tf.fill([5,5], self.u[2])
        U1 = tf.linalg.set_diag(U1, tf.repeat(tf.expand_dims(self.u[3], 0), 5))
        U2 = tf.fill([5,5], self.u[4])
        U2 = tf.linalg.set_diag(U2, tf.repeat(tf.expand_dims(self.u[5], 0), 5))
        U3 = tf.fill([5,5], self.u[6])
        U3 = tf.linalg.set_diag(U3, tf.repeat(tf.expand_dims(self.u[7], 0), 5))
        self.A = tf.concat( [tf.concat([U, U1], axis=1), tf.concat([U2, U3], axis=1)], axis=0)

        return tf.nn.gelu(tf.matmul(inputs, self.A))


class permLayerVariable(tf.keras.layers.Layer):

    """
    Custom layer which is equivariant with respect to the permutation symmetry of the Fermat quintic. This uses the parameter sharing scheme, but additionally allows for variable output size. However, this contruction is already conceptually flawed and can be disregarded for now.
    """

    def __init__(self, shape):
        super(permLayerVariable, self).__init__()
        self.shape = shape
        self.u = self.add_weight(
            shape=(2*self.shape[0]*self.shape[1],), initializer="random_normal", trainable=True
        ) 


    def call(self, inputs):

        k = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if j == 0:
                    U = tf.fill([5,5], self.u[k])
                    k += 1
                    U = tf.linalg.set_diag(U, tf.repeat(tf.expand_dims(self.u[k],0), 5))
                    k += 1
                else:
                    V = tf.fill([5,5], self.u[k])
                    k += 1
                    V = tf.linalg.set_diag(V, tf.repeat(tf.expand_dims(self.u[k], 0), 5))
                    k += 1
                    U = tf.concat([U,V], axis=1)
             
            if i == 0:
                self.A = U
            else:
                self.A = tf.concat([self.A, U], axis=0)

        return tf.nn.gelu(tf.matmul(inputs, self.A))
    

"""
TEST SNIPPET

This is used to check shapes etc.
"""
# x = tf.expand_dims( tf.range(10, dtype=tf.float32), axis=0 )

# test = simpleScaleConv()

# y = test(x)