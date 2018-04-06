from road import RoadNode
from trip import Trip
from database_access import DatabaseAccess
from grid import GridCell

from pandas import DataFrame



def initCity(create_trips = False,map_trip_to_cell=False,map_trip_to_road = False):
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    # create roads
    RoadNode.setDao(dao)
    RoadNode.createRoads()
    # create grids
    GridCell.setDao(dao)
    GridCell.initAllCells(n_grids=20)
    GridCell.mapRoadsToCell()


    Trip.setDao(dao)
    if create_trips:
        Trip.createAllTrips(save_pickle=True)
    elif map_trip_to_cell:
        # map each gps timestamp to a grid
        Trip.getTripsPickle()
        Trip.mapTripToCell(coord_dict=GridCell.cell_coord_dict,
                       lat_cut=GridCell.lat_cut_points,
                       lon_cut=GridCell.lon_cut_points,
                       save_pickle=True)
    elif map_trip_to_road:
        Trip.getTripsPickle()
        Trip.mapTripToRoads(cell_dict=GridCell.cell_id_dict,save_pickle=True)
    else:
        Trip.getTripsPickle()

















if __name__ == '__main__':
    initCity()
    m = Trip.computeTransitionMatrices(hops=[1])

    DataFrame(m[0]).to_csv("/Volumes/Porter's Data/penn-state/data-sets/tmp.csv")
    tmp = 0
