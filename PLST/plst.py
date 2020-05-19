import numpy as np
from numpy import linalg as la
from sklearn.linear_model import Ridge
from sklearn.preprocessing import minmax_scale

class PLST():
    def __init__(self, m, alpha=0.1):
        '''init

         Parameters
        ----------
        m:      label space compressed dimension, less than label number
        alpha:  linear regression regular coefficient
        '''
        self.m = m
        self.alpha = alpha
        self.models = []

    def fit(self, X, y):
        '''fit model

         Parameters
        ----------
        X:  numpy.ndarray
            train input feature
        y:  numpy.ndarray {0,1}
            train output
        '''
        y_new = np.copy(y)
        y_new[y_new == 0] = -1
        z, self.Um = self.encode(y_new)
        new_X = np.c_[np.ones(X.shape[0]), X]
        # regress x on z 
        z = z.T
        for i in range(len(z)):
            current_y = z[i]
            #print(current_y)
            #print(current_y.shape)
            linear_regress = Ridge()
            linear_regress.fit(new_X,current_y)
            self.models += [linear_regress]
        print('train complete')
        return self

    def encode(self, y):
        '''encode y use svd

         Parameters
        ----------
        y:  numpy.ndarray {0,1}
            train output of shape :code:`(n_samples, n_target)`

        Returns
        -------
        z:      numpy.ndarray
                dimensionality reduction matrix of y shape :code:`(n_samples, m)`
        Vm:     numpy.ndarray
                top mright singular matrix after svd shape :code:`(n_features, m)`
        shift:  numpy.ndarray
                mean of y by col shape :code:`(1, n_features)`
        '''
        y = y.T
        #shift = np.mean(y, axis=0)
        #y_shift = y - shift
        U, var, _= la.svd(y)
        # u is shape n_sample * m
        this_var = sum(var[i] for i in range(self.m))
        print('variance accounted for m = {} is {}'.format(self.m,this_var/ sum(var)))
        Um = U[:,0:self.m]
        #print(y_shift)
        #print(Vm)
        y = y.T
        z = np.dot(y,Um)
        return z, Um

    def predict(self, X):
        '''encode y use svd

        Parameters
       ----------
       X:   numpy.ndarray
            train input feature :code:`(n_samples, n_features)`

       Returns
       -------
       y_pred:      numpy.ndarray {0, 1}
                    predict of y shape :code:`(n_samples, n_traget)`
       y_pred_prob: numpy.ndarray [0, 1]
                    predict probility of y  shape :code:`(n_features, n_traget)`
        '''
        
        new_X = np.c_[np.ones(X.shape[0]), X]
        result = np.zeros((X.shape[0],self.Um.shape[0]))
        for idx, x in enumerate(new_X):
            this_result = np.zeros(self.Um.shape[0])
            for i,model in enumerate(self.models):
                Um = self.Um.T
                current_U = Um[i]
                pred = model.predict(x.reshape(1,-1))
                this_result += pred * current_U
            result[idx] = this_result
            
        y_pred_prob = minmax_scale(result, axis=1)
        y_pred = np.zeros(result.shape)
        y_pred[result > 0] = 1
        y_pred[result <= 0] = 0
        
        '''
        z_pred = self.w * np.c_(np.ones(X.shape(0)), X)
        y_real = z_pred * self.Vm.T + self.shift
        y_pred = np.zeros(y_real.shape)
        y_pred[y_real > 0] = 1
        y_pred[y_real <= 0] = 0
        y_pred_prob = minmax_scale(y_real, axis=1)
        '''
        return y_pred, y_pred_prob


class PLST_tree():
    def __init__(self, m, alpha=0.1):
        '''init

         Parameters
        ----------
        m:      label space compressed dimension, less than label number
        alpha:  linear regression regular coefficient
        '''
        self.m = m
        self.alpha = alpha
        self.models = []

    def fit(self, X, y):
        '''fit model

         Parameters
        ----------
        X:  numpy.ndarray
            train input feature
        y:  numpy.ndarray {0,1}
            train output
        '''
        y_new = np.copy(y)
        y_new[y_new == 0] = -1
        z, self.Um = self.encode(y_new)
        new_X = np.c_[np.ones(X.shape[0]), X]
        # regress x on z 
        z = z.T
        for i in range(len(z)):
            current_y = z[i]
            #print(current_y)
            #print(current_y.shape)
            tree_regress = DecisionTreeRegressor()
            tree_regress.fit(new_X,current_y)
            self.models += [tree_regress]
        print('train complete')
        return self

    def encode(self, y):
        '''encode y use svd

         Parameters
        ----------
        y:  numpy.ndarray {0,1}
            train output of shape :code:`(n_samples, n_target)`

        Returns
        -------
        z:      numpy.ndarray
                dimensionality reduction matrix of y shape :code:`(n_samples, m)`
        Vm:     numpy.ndarray
                top mright singular matrix after svd shape :code:`(n_features, m)`
        shift:  numpy.ndarray
                mean of y by col shape :code:`(1, n_features)`
        '''
        y = y.T
        #shift = np.mean(y, axis=0)
        #y_shift = y - shift
        U, var, _= la.svd(y)
        # u is shape n_sample * m
        this_var = sum(var[i] for i in range(self.m))
        print('variance accounted for m = {} is {}'.format(self.m,this_var/ sum(var)))
        Um = U[:,0:self.m]
        #print(y_shift)
        #print(Vm)
        y = y.T
        z = np.dot(y,Um)
        return z, Um

    def predict(self, X):
        '''encode y use svd

        Parameters
       ----------
       X:   numpy.ndarray
            train input feature :code:`(n_samples, n_features)`

       Returns
       -------
       y_pred:      numpy.ndarray {0, 1}
                    predict of y shape :code:`(n_samples, n_traget)`
       y_pred_prob: numpy.ndarray [0, 1]
                    predict probility of y  shape :code:`(n_features, n_traget)`
        '''
        
        new_X = np.c_[np.ones(X.shape[0]), X]
        result = np.zeros((X.shape[0],self.Um.shape[0]))
        for idx, x in enumerate(new_X):
            this_result = np.zeros(self.Um.shape[0])
            for i,model in enumerate(self.models):
                Um = self.Um.T
                current_U = Um[i]
                pred = model.predict(x.reshape(1,-1))
                this_result += pred * current_U
            result[idx] = this_result
            
        y_pred_prob = minmax_scale(result, axis=1)
        y_pred = np.zeros(result.shape)
        y_pred[result > 0] = 1
        y_pred[result <= 0] = 0
        
        '''
        z_pred = self.w * np.c_(np.ones(X.shape(0)), X)
        y_real = z_pred * self.Vm.T + self.shift
        y_pred = np.zeros(y_real.shape)
        y_pred[y_real > 0] = 1
        y_pred[y_real <= 0] = 0
        y_pred_prob = minmax_scale(y_real, axis=1)
        '''
        return y_pred, y_pred_prob




