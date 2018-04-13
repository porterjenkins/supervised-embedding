import pickle
from database_access import DatabaseAccess
import numpy as np

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
        y = np.zeros(shape = (len(cls.all_roads),1))
        lane_vector = np.zeros(shape = (len(cls.all_roads),1))


        W = np.zeros_like(similarity_matrix)
        for i, road in enumerate(cls.all_roads.values()):
            for j, neighbor in enumerate(road.adjacent_roads):
                W[i,neighbor.node_id] = similarity_matrix[i,neighbor.node_id]

            volume = 0
            for camera in road.cameras:
                volume += camera.volume
            y[i] = volume

            n_lanes_i = road.lane_cnt
            lane_vector[i, 0] = n_lanes_i

        F_g = np.matmul(W,y)

        # Concatenate F_g and lanes to get X
        X = np.concatenate((lane_vector,F_g),axis=1)

        return X, y




if __name__ == '__main__':
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    RoadNode.setDao(dao)
    RoadNode.createRoads()
    RoadNode.getRoadFeatureMatrix()