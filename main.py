from road import RoadNode
from trip import Trip
from database_access import DatabaseAccess
from grid import GridCell
from camera import TrafficCam
from model import Model
from sklearn.model_selection import train_test_split
from parser import GraphParser

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
    #parser = GraphParser(dao)
    #parser.graphParser(xml_file="jinan_large.osm")
    initCity(dao=dao,trip_pickle=True,cam_pickle=True)
    transition = Trip.computeTransitionMatrices(hops=[1],l2_norm=True)
    X, y = RoadNode.getRoadFeatures(similarity_matrix=transition)
    print(X)
    print(y)

    train_idx, test_idx = train_test_split(range(len(y)))

    model_transition = Model(X=X,y=y,similarity_mtx=transition)
    model_transition.regression(train_idx=train_idx,test_idx=train_idx,regression_method='OLS')