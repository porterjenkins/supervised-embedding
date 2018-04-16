from database_access import DatabaseAccess
import numpy as np
from sklearn.linear_model import LinearRegression



class Model:

    def __init__(self,X,y,similarity_mtx):
        self.edge_feature_mat = X
        self.edge_volume_mat = y
        self.similarity_mtx = similarity_mtx

    @staticmethod
    def eval(y, y_pred, thre = 5):
        if y.shape != y_pred.shape:
            print('Inequal shape!')
            return -1
        mask = y>thre
        rmse = np.sqrt(np.mean(np.square(y-y_pred)))
        mape = np.mean(np.abs(y[mask]-y_pred[mask])/y[mask])
        return rmse, mape

    def lP(self, train_idx, test_idx, max_iter=1000):
        result = np.copy(self.map.edge_volume_mat)
        norm_graph = np.transpose(self.map.graph)
        norm_sum = norm_graph.sum(axis=0)
        norm_sum[norm_sum == 0] = 1
        norm_graph = norm_graph / norm_sum
        norm_graph = np.transpose(norm_graph)
        for iter in range(max_iter):
            result = np.dot(norm_graph, result)
            result[train_idx] = self.map.edge_volume_mat[train_idx]
        rmse, mape = Model.eval(self.map.edge_volume_mat[test_idx], result[test_idx])
        print("test lp: rmse {0}, mape {1}.".format(rmse, mape))
        ipdb.set_trace()

    def typeAvg(self, train_idx, test_idx):
        sums = np.zeros((15, self.map.edge_volume_mat.shape[1]))
        cnts = np.zeros((15))
        for train_id in train_idx:
            type = int(self.map.edge_feature_mat[train_id][-1])
            # ipdb.set_trace()
            sums[type] += self.map.edge_volume_mat[train_id]
            cnts[type] += 1
        # ipdb.set_trace()
        cnts[cnts == 0] = 1
        for type in range(15):
            sums[type] /= cnts[type]
        y_pred = np.zeros_like(self.map.edge_volume_mat)
        for test_id in test_idx:
            test_type = int(self.map.edge_feature_mat[test_id][-1])
            y_pred[test_id] = sums[test_type]

        rmse, mape = Model.eval(self.map.edge_volume_mat[test_idx], y_pred[test_idx])
        print("test avg: rmse {0}, mape {1}.".format(rmse, mape))
        # ipdb.set_trace()

    def regression(self, train_idx, test_idx, regression_method):
        #y_pred = np.zeros(shape=self.edge_volume_mat.shape[0])
        #y_pred = np.zeros(self.edge_volume_mat[test_idx])
        #for t in range(self.edge_volume_mat.shape[1]):
        #    if t % 100 is 0:
                #print(t)
        model = LinearRegression()
        model.fit(self.edge_feature_mat[train_idx], self.edge_volume_mat[train_idx])
        y_pred = model.predict(self.edge_feature_mat[test_idx])
        rmse, mape = Model.eval(self.edge_volume_mat[test_idx], y_pred)
        print("test regression {2}: rmse {0}, mape {1}.".format(rmse, mape, regression_method))