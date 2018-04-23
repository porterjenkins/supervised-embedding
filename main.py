from road import RoadNode
from trip import Trip
from database_access import DatabaseAccess
from grid import GridCell
from camera import TrafficCam
from model import Model
from sklearn.model_selection import train_test_split

import numpy as np

def initCity(dao,trip_pickle = True,cam_pickle=True):
    """
    Initialize all data for following object:
        - Roads
        - Grids
        - Trips
        - Cameras
    Link/map data:
        - Trips --> Cells
        - Trip/Cells --> Roads
        - Cameras --> Roads
        - Roads --> Cameras

    :param read_pickle: If true, read ALL data from pickle. Otherwise initialize from scratch: can be slow
    :return:
    """

    # Initialize roads
    RoadNode.init(dao)
    # Initialize grids
    GridCell.init(dao)
    # Initialize Trips
    Trip.init(dao,read_pickle=trip_pickle)

    # create traffic cameras
    TrafficCam.init(dao,read_pickle=cam_pickle)








if __name__ == '__main__':
    # Initialize DatabaseAcessObject (dao)
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    initCity(dao=dao,trip_pickle=True,cam_pickle=True)


    # save matrices for xianfeng's code
    """fname = "/Users/porterjenkins/Documents/PENN STATE/RESEARCH/supervised-embedding/xianfeng/city-eye/data_back/"
    monitored_roads = RoadNode.getMonitoredRoads()
    np.savez(fname + "monitored_file-porter.npz", monitored_nodes = monitored_roads)
    rawnodes = RoadNode.getNodeVolumeMatrix()
    np.savez(fname + "nodes-porter.npz",nodes = rawnodes)
    adjacency = RoadNode.getAdjacencyMatrix(tensor=True)
    np.savez(fname + "weights-porter.npz", weights = adjacency)
    transition = Trip.computeTransitionMatrices(hops=[5],l2_norm=False)
    np.savez(fname + "flows-porter.npz", flows = transition)"""


    #transition = Trip.computeTransitionMatrices(hops=[10], l2_norm=True)
    transition = RoadNode.getGraphSimilarityMtx(method="euclidean")

    X, y = RoadNode.getRoadFeatures(similarity_matrix=transition,n_ts=24,filter_neighbors=False)
    train_idx, test_idx = train_test_split(range(len(y)))

    model_transition = Model(X=X,y=y,similarity_mtx=transition)
    model_transition.regression(train_idx,test_idx)


