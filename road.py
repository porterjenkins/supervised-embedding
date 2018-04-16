import pickle
from database_access import DatabaseAccess
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import OneHotEncoder

class RoadNode:

    # Class attributes
    all_roads = dict()
    dao = None

    def __init__(self,node_id,coordinates,adjacent_roads,lane_cnt):
        self.node_id = node_id
        self.coordinates = coordinates
        self.adjacent_roads = adjacent_roads
        self.lane_cnt = lane_cnt
        self.grid_cell = None
        self.cameras = list()

    def __str__(self):
        return self.node_id

    def updateAdjacencyList(self,new_list):
        self.adjacent_roads = new_list

    def updateGridCell(self,grid_cell):
        self.grid_cell = grid_cell

    def addCamera(self,new_cam):
        self.cameras.append(new_cam)

    def getCameraVolume(self):
        arr_list = []
        if self.cameras:
            for c, camera in enumerate(self.cameras):
                volume_samples = list(camera.volume.values())
                if volume_samples:
                    arr_list.append(np.array(volume_samples,dtype='float64').reshape(-1,1))

            if arr_list:
                volume_matrix = np.concatenate(arr_list,axis=1)
                volume = np.sum(volume_matrix,axis=1)


            else:
                volume = None

        else:
            volume = None



        return volume



    @classmethod
    def setDao(cls,dao):
        cls.dao = dao

    @classmethod
    def allRoadsToPickle(cls):
        fname = cls.dao.road_dir + "/pickle/all_roads.p"
        pickle.dump(cls.all_roads, open(fname, 'wb'))

    @classmethod
    def getRoadsPickle(cls):
        print("Retrieving road network from pickle...")
        fname = cls.dao.road_dir + "/pickle/all_roads.p"
        cls.all_roads = pickle.load(open(fname,'rb' ))
        print("{} roads found".format(len(cls.all_roads)))


    @classmethod
    def updateAllAdjacencies(self,all_roads):
        all_roads_new = {}
        for road in all_roads.values():
            adj_list_new = []
            for adj_road in road.adjacent_roads:
                tmp = all_roads[adj_road]
                adj_list_new.append(tmp)
            road.updateAdjacencyList(adj_list_new)
            all_roads_new[road.node_id] = road
        return all_roads_new

    @classmethod
    def createRoads(cls):
        print("Creating road network...")

        edges = cls.dao.getEdgeList()
        nodes = cls.dao.getNodeCoordinates()
        lanes = cls.dao.getNodeLanes()


        for node in nodes.itertuples():
            edge_list = edges[edges.node == node.node]
            adjacent_list = []
            for row in edge_list.itertuples():
                adjacent_list.append(row.adj_node)
            n_lanes_i = lanes.loc[row.node].values[0]
            road_i = RoadNode(node_id=node.node,
                                coordinates=[node.latitude,node.longitude],
                                adjacent_roads = adjacent_list,
                                lane_cnt= n_lanes_i)
            cls.all_roads[node.node] = road_i
        cls.all_roads = cls.updateAllAdjacencies(cls.all_roads)
        print("{} roads found".format(len(cls.all_roads)))


    @classmethod
    def init(cls,dao):
        RoadNode.setDao(dao)
        RoadNode.createRoads()

    @classmethod
    def getRoadFeatures(cls,similarity_matrix,n_neighbors = 2):
        print("\nGetting road features...")

        n_roads = len(cls.all_roads)
        n_date_times = 24
        y = np.zeros(shape=(n_roads,n_date_times))
        lanes_list = list()
        time_feature_list = list()


        W = np.zeros_like(similarity_matrix)
        for i, road in enumerate(cls.all_roads.values()):
            for j, neighbor in enumerate(road.adjacent_roads):
                W[i,neighbor.node_id] = similarity_matrix[i,neighbor.node_id]

            volume = road.getCameraVolume()
            if volume is not None:
                y[i,:] = volume

            n_lanes_i = np.array([road.lane_cnt] * n_date_times)
            lanes_list.append(n_lanes_i)

            # insert indicators for time of day
            time = np.array(range(1,n_date_times+1))
            enc = OneHotEncoder()
            time_mtx_i = enc.fit_transform(X=time.reshape(-1,1)).toarray()
            time_feature_list.append(time_mtx_i)


        lane_vector = np.concatenate(lanes_list)
        time_mtx = np.concatenate(time_feature_list,axis=0)

        X_list = []
        for j in range(n_date_times):
            F_g_date = np.matmul(W, y[:,j])
            X_list.append(F_g_date)

        F_g = np.concatenate(X_list)
        # Concatenate F_g and lanes to get X
        features = (lane_vector.reshape(-1,1),F_g.reshape(-1,1),time_mtx)
        X = np.concatenate(features,axis=1)
        y = y.flatten().reshape(-1,1)


        return X, y

    @classmethod
    def getGraphSimilarityMtx(cls):
        print("Computing graph similarity")
        n_roads = len(cls.all_roads)
        similarity_mtx = np.zeros(shape = (n_roads,n_roads))

        """cnt = 0
        for i, road_i in cls.all_roads.items():
            for j, road_j in cls.all_roads.items():
                cnt += 1
                if i !=j:
                    similarity_mtx[i,j] = 1/(euclidean(road_i.coordinates,road_j.coordinates))
                else:
                    similarity_mtx[i, j] = 1

                progress = round((cnt / float(n_roads**2)) * 100, 2)

                sys.stdout.write("\r Getting graph similarity matrix: {}% complete".format(progress))
                sys.stdout.flush()
                """

        cnt = 0
        i = 0
        while i < n_roads:
            j = i + 1
            road_i = cls.all_roads[i]
            while j < n_roads:
                road_j = cls.all_roads[j]
                similarity_mtx[i, j] = 1 / (euclidean(road_i.coordinates, road_j.coordinates))
                j += 1

            i += 1

        mtx_t = np.transpose(similarity_mtx)
        similarity_mtx = similarity_mtx + mtx_t

        return similarity_mtx






if __name__ == '__main__':
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    RoadNode.setDao(dao)
    RoadNode.createRoads()
    RoadNode.getRoadFeatureMatrix()