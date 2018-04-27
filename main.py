from road import RoadNode
from trip import Trip
from database_access import DatabaseAccess
from grid import GridCell
from camera import TrafficCam
from model import Model
from sklearn.model_selection import train_test_split
import numpy as np

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
    dao = DatabaseAccess(city='jinan',
                         data_dir="/Volumes/Porter's Data/penn-state/data-sets/",
                         lat_range= (36.6467,36.6738), # filter city to smaller window
                         lon_range= (116.9673,117.0286))

    initCity(dao=dao,trip_pickle=True,cam_pickle=True)


    """

    # save matrices for xianfeng's code
    fname = "/Users/porterjenkins/Documents/PENN STATE/RESEARCH/supervised-embedding/xianfeng/city-eye/data_back/"
    monitored_roads = RoadNode.getMonitoredRoads()
    np.savez(fname + "monitored_file-porter-small.npz", monitored_nodes = monitored_roads)
    rawnodes = RoadNode.getNodeVolumeMatrix()
    np.savez(fname + "nodes-porter-small.npz",nodes = rawnodes)
    adjacency = RoadNode.getAdjacencyMatrix(tensor=True)
    #adjacency[0] = np.transpose(adjacency[0])
    np.savez(fname + "weights-porter-small.npz", weights = adjacency)
    #transition = Trip.computeTransitionMatrices(hops=range(1,6),l2_norm=True)
    #transition = RoadNode.getEmbeddingSimilarity("road_embedding_50.pickle",l2_norm=True)
    np.savez(fname + "flows-porter-small.npz", flows = transition)
    """




    #sim_mtx = RoadNode.getEmbeddingSimilarity("road_embedding_50.pickle",l2_norm=True)
    #sim_mtx = RoadNode.getGraphSimilarityMtx(method="euclidean")
    sim_mtx = Trip.computeTransitionMatrices([1], l2_norm=True)

    #sim_mtx = Model.takeTopK(sim_mtx,k=5)

    X, y = RoadNode.getRoadFeatures(similarity_matrix=sim_mtx,n_ts=24,filter_neighbors=False)
    model_transition = Model(X=X, y=y, similarity_mtx=sim_mtx,n_ts=24,n_road=len(RoadNode.all_roads))
    monitored_roads = RoadNode.getMonitoredRoads()
    train_idx, test_idx = model_transition.testTrainSplit(test_pct=.2,monitored_roads=monitored_roads,set_seed=123)

    model_transition.regression(train_idx,test_idx)


