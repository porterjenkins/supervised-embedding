import pickle
from database_access import DatabaseAccess
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import OneHotEncoder
from collections import deque, OrderedDict


class RoadNode:

    # Class attributes
    all_roads = dict()
    dao = None
    path = dict() # used for BFS search for shortest path

    def __init__(self,node_id,coordinates,adjacent_roads,lane_cnt):
        self.node_id = node_id
        self.coordinates = coordinates
        self.adjacent_roads = adjacent_roads
        self.lane_cnt = lane_cnt
        self.grid_cell = None
        self.cameras = list()
        self.marked = False # used for BFS search for shortest path

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
                volume = np.max(volume_matrix,axis=1)


            else:
                volume = None

        else:
            volume = None



        return volume

    @classmethod
    def bfsSearch(cls,root):
        # Breadth-first search (BFS)
        q = deque()
        root.marked = True
        q.append(root)

        while len(q) > 0:
            r = q.pop()

            for neighbor in r.adjacent_roads:
                if not neighbor.marked:
                    neighbor.marked = True
                    q.append(neighbor)
                    cls.path[neighbor.node_id] = r.node_id




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
        #lanes = cls.dao.getNodeLanes()
        from pandas import DataFrame
        lanes = DataFrame(np.ones(shape = (len(nodes),1)))


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
    def getRoadFeatures(cls,similarity_matrix,n_date_times=24):
        print("\nGetting road features...")

        p = 1
        n_roads = len(cls.all_roads)
        y_mtx = np.zeros(shape=(n_roads,n_date_times))
        #X = np.zeros(shape = (n_date_times,n_roads,p))
        X = OrderedDict()
        y = OrderedDict()
        lanes = np.zeros(shape=(n_roads,n_date_times))
        time_feature_list = list()

        for i, road in enumerate(cls.all_roads.values()):
            volume = road.getCameraVolume()
            if volume is not None:
                y_mtx[i,:] = volume

            #lanes[i,:] = np.array([road.lane_cnt] * n_date_times)




        #W = np.zeros_like(similarity_matrix)
        """for i, road in enumerate(cls.all_roads.values()):
            # Hard threshold on W: only consider direct neighbors, else 0
            #for j, neighbor in enumerate(road.adjacent_roads):
            #    W[i,neighbor.node_id] = similarity_matrix[i,neighbor.node_id]

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
            F_g_date = np.matmul(similarity_matrix, y[:,j])
            X_list.append(F_g_date)
            


        F_g = np.concatenate(X_list)
        # Concatenate F_g and lanes to get X
        features = (lane_vector.reshape(-1,1),F_g.reshape(-1,1),time_mtx)
        X = np.concatenate(features,axis=1)
        y = y.flatten().reshape(-1,1)
        """
        for j in range(n_date_times):
            X[j] = np.zeros(shape=(n_roads, p))
            y[j] = np.zeros(shape=n_roads)
            F_g_date = np.matmul(similarity_matrix, y_mtx[:, j])
            X[j][:,0] = F_g_date
            #X[j][:,1] = lanes[:,j]


            # Remove samples where y_ij == 0 --> Treat 0 as missing value
            keep_idx = np.where(y_mtx[:,j] > 0)
            X[j] = X[j][keep_idx[0],:]
            y[j] = y_mtx[keep_idx[0],j]





        return X, y

    @classmethod
    def getGraphSimilarityMtx(cls,method='euclidean'):
        print("Computing graph similarity")
        n_roads = len(cls.all_roads)


        similarity_mtx = np.zeros(shape = (n_roads,n_roads))

        if method == 'euclidean':
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
        elif method == 'shortest_path':
            root = cls.all_roads[0]
            cls.bfsSearch(root)

            #shortest_path
            s = cls.all_roads[0]
            w = cls.all_roads[5]
            path = []

            current_node = w
            parent_node = cls.path[w.neighbor.node_id]

            while current_node.node_id !=  parent_node.node_id:
                current_node = parent_node
                parent_node = cls.path[current_node.node_id]
                path.append(current_node)


        else:
            raise Exception("Method must be a member of ['euclidean','shortest_path']")

        return similarity_mtx

    @classmethod
    def getNodeVolumeMatrix(cls,n_ts=24):

        n_roads = len(cls.all_roads)

        node_mtx = np.zeros(shape = (n_roads,n_ts))

        for id, road in cls.all_roads.items():
            if road.cameras:
                tmp = 0
            volume = road.getCameraVolume()
            if volume is not None:
                node_mtx[id,:] = volume


        return node_mtx

    @classmethod
    def getAdjacencyMatrix(cls,tensor):

        n_roads = len(cls.all_roads)
        if tensor:
            A = np.zeros(shape=(1,n_roads,n_roads))
        else:
            A = np.zeros(shape=(n_roads, n_roads))

        for i, road_i in cls.all_roads.items():
            for road_j in road_i.adjacent_roads:
                j = road_j.node_id
                if tensor:
                    A[0,i,j] = 1
                else:
                    A[i,j] = 1

        return A

    @classmethod
    def getMonitoredRoads(cls):
        roads_w_cam = list()
        for i, road in cls.all_roads.items():
            if road.cameras:
                roads_w_cam.append(i)

        return np.array(roads_w_cam)










if __name__ == '__main__':
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    RoadNode.setDao(dao)
    RoadNode.createRoads()