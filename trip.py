from database_access import DatabaseAccess
from road import RoadNode
from grid import GridCell
import numpy as np
import pickle
from scipy.spatial.distance import euclidean
import sys
from sklearn.preprocessing import normalize

class Trip(object):

    # class attributes
    all_trips = None
    dao = None


    def __init__(self,device_sn,date,trajectory):
        self.device_sn = device_sn
        self.date = date
        self.trajectory = trajectory


    def updateRoadNodes(self,node_list):
        self.trajectory['road_node'] = node_list


    def updateCellList(self,cell_list):
        """
        - Add a list of cell id's at each gps timestamp to the trajectory DataFrame
        - Drop all records where cell id is null:
            - This ignores gps timestamps that are observed outside the ciy window
        :param cell_list: (list) list of cell ids to be added to dataframe
        :return:
        """
        self.trajectory['cell'] = cell_list
        # Drop all gps timestamps w/o cell assignment
        self.trajectory = self.trajectory[-self.trajectory.cell.isnull()]


    @classmethod
    def setDao(cls,dao):
        cls.dao = dao

    @classmethod
    def allTripsToPickle(cls):
        fname = cls.dao.traj_dir + "/pickle/all_trajectory_small.p"
        pickle.dump(cls.all_trips, open(fname, 'wb'),protocol=2)

    @classmethod
    def createAllTrips(cls,save_pickle=False):
        print("Creating trips...")
        files = cls.dao.getTrajFiles()
        dates = list(files.keys())
        all_trips = []
        for date in dates:
            for f in files[date]:
                if 'gps' in f:
                    gps_dta = cls.dao.readGpsData(date,f)
                    if gps_dta is not None:
                        trip = Trip(device_sn=f.split("_")[0],date=date,trajectory=gps_dta)
                        all_trips.append(trip)
                elif 'obd':
                    pass

        cls.all_trips = all_trips
        print("{} trips found".format(len(cls.all_trips)))
        if save_pickle:
            cls.allTripsToPickle()


    @classmethod
    def getTripsPickle(cls):
        print("Retrieving trips from pickle...")
        fname = cls.dao.traj_dir + "/pickle/all_trajectory_small.p"
        all_trips = pickle.load(open(fname,'rb' ))
        cls.all_trips = all_trips
        print("{} trips found".format(len(cls.all_trips)))

        n_gps_points = 0
        for trip in cls.all_trips:
            n_gps_points += len(trip.trajectory)

        print("{} total gps records founds".format(n_gps_points))


    @classmethod
    def mapTripToCell(cls,coord_dict,lat_cut,lon_cut,save_pickle=False):
        """
        Here we map each gps timestamp to a cell in city-wide grid.
            - For each gps timestamp we use the lat/lon coordinates to look up the corresponding cell id using the
                dictionary, coord_dict
            - Note: not all gps timestamps will successfully map to a cell. This occurs when a given gps coordcinate
                is observed OUTSIDE of the designated city window
        :param coord_dict: (dict) Dictionary that maps cell coordinate bounds to cell id
                    - keys (tuple): ((lat_lower_bound, lat_upper_bound), (lon_lower_bound,lon_upper_bound))
                    - values (int): cell id
        :param lat_cut: (list) list of cut points that discretize latitude range
        :param lon_cut: (list) list of cut points that discretize longitude range
        :return:
        """


        for trip in Trip.all_trips:
            cell_list = []
            for time_stamp in trip.trajectory.itertuples():
                ts_loc = (time_stamp.latitude, time_stamp.longitude)

                # search latitude
                for i in range(len(lat_cut)-1):
                    lat_lower = lat_cut[i]
                    lat_upper = lat_cut[i+1]
                    if ts_loc[0] >= lat_lower and ts_loc[0] < lat_upper:
                        lat_key = (lat_lower,lat_upper)
                        break
                    else:
                        lat_key = None

                # search longitude
                for i in range(len(lon_cut) - 1):
                    lon_lower = lon_cut[i]
                    lon_upper = lon_cut[i + 1]
                    if ts_loc[1] >= lon_lower and ts_loc[1] < lon_upper:
                        lon_key = (lon_lower, lon_upper)
                        break
                    else:
                        lon_key = None

                if lat_key is not None and lon_key is not None:
                    ts_cell = coord_dict[(lat_key,lon_key)]
                    cell_list.append(ts_cell)
                else:
                    # either lat or lon measurement resides outside city window
                    cell_list.append(np.nan)

            trip.updateCellList(cell_list)

        if save_pickle:
            cls.allTripsToPickle()

    @classmethod
    def mapTripToCell2(cls,coord_dict,lat_cut,lon_cut,save_pickle=False):
        print(" - Mapping roads to cells")

        lat_dist = lat_cut[1] - lat_cut[0]
        lon_dist = lon_cut[1] - lon_cut[0]

        trip_cnt = 0
        for trip in Trip.all_trips:
            trip_cnt += 1
            cell_list = []
            for time_stamp in trip.trajectory.itertuples():

                # search latitude
                # check if in window
                if time_stamp.latitude >= lat_cut[0] and time_stamp.latitude < lat_cut[-1:][0]:
                    lat_key = int((time_stamp.latitude - lat_cut[0]) // lat_dist)

                else:
                    lat_key = None


                # search longitude
                if time_stamp.longitude >= lon_cut[0] and time_stamp.longitude < lon_cut[-1:][0]:
                    lon_key = int((time_stamp.longitude - lon_cut[0]) // lon_dist)

                else:
                    lon_key = None

                if lat_key is not None and lon_key is not None:
                    ts_cell = coord_dict[(lat_key, lon_key)]
                    cell_list.append(ts_cell)
                else:
                    # either lat or lon measurement resides outside city window
                    cell_list.append(np.nan)
            trip.updateCellList(cell_list)
            progress = round((trip_cnt / float(len(Trip.all_trips))) * 100, 2)

            sys.stdout.write("\r Trip mapping progress: {}% of trips complete".format(progress))
            sys.stdout.flush()
        print(" - Complete...")
        if save_pickle:
            cls.allTripsToPickle()


    @classmethod
    def mapTripToRoads(cls,cell_dict,save_pickle=False):
        print(" -Mapping trips to roads...")
        trip_cnt = 0
        n_trips = len(Trip.all_trips)
        for trip in Trip.all_trips:
            trip_cnt += 1
            clostest_road_list = []
            for time_stamp in trip.trajectory.itertuples():
                cell_i = cell_dict[time_stamp.cell]
                car_road_dist = []

                if not cell_i.road_list:
                    # if road list is empty then append null - ie., no road found
                    clostest_road_list.append(np.nan)
                else:
                    for road in cell_i.road_list:
                        car_road_dist.append(euclidean((time_stamp.latitude,time_stamp.longitude), road.coordinates))

                    closest_road = cell_i.road_list[np.argmin(car_road_dist)]
                    clostest_road_list.append(closest_road.node_id)

            trip.updateRoadNodes(clostest_road_list)

            progress = round((trip_cnt / float(n_trips))*100,2)

            sys.stdout.write("\r Trip mapping progress: {}% of trips complete".format(progress))
            sys.stdout.flush()

        print(" -Complete...")

        if save_pickle:
            cls.allTripsToPickle()

    @classmethod
    def init(cls,dao,read_pickle):
        Trip.setDao(dao)
        if read_pickle:
            Trip.getTripsPickle()
        else:
            Trip.createAllTrips(save_pickle=True)
            print("Mapping roads...")
            Trip.mapTripToCell2(coord_dict=GridCell.cell_coord_dict,
                               lat_cut=GridCell.lat_cut_points,
                               lon_cut=GridCell.lon_cut_points,
                               save_pickle=True)
            Trip.mapTripToRoads(cell_dict=GridCell.cell_id_dict, save_pickle=True)



    @classmethod
    def computeTransitionMatrices(cls,hops,l2_norm = True):
        """

        :param hops: (list) hop lengths to try; i.e., [1,2,3,10]
        :return: (n-d array)
        """
        print("\n Calculating transition matrices with hops: {}".format(hops))
        n_roads = len(RoadNode.all_roads)

        # initialize road tensor
        transition = np.zeros(shape=[len(hops),n_roads,n_roads])

        cnt = 0
        for trip in Trip.all_trips:

            roads_unique = trip.trajectory.road_node.unique()

            # check if trajectory has at least two points (starting and ending road segments)
            if len(roads_unique) >=2:
                hop_cnt = 0
                for hop_length in hops:
                    cnt += 1
                    for road_idx in range(len(roads_unique) - hop_length):

                        try:
                            start_road = int(roads_unique[road_idx])
                            end_road = int(roads_unique[road_idx + hop_length])
                            transition[hop_cnt, RoadNode.node_to_mtx_idx[start_road], RoadNode.node_to_mtx_idx[end_road]] += 1
                        except ValueError:
                            # Allow function to be robust to the hop length...
                            pass


                    hop_cnt +=1

                    progress = round((cnt / float(len(Trip.all_trips) * len(hops))) * 100, 2)
                    sys.stdout.write("\r Transition Matrices --> {}% complete".format(progress))
                    sys.stdout.flush()

        # sum of over hops; i.e., M = A + A^2 + A^3 .... A^n
        transition = transition.sum(axis=0)

        sys.stdout.write("\r Transition Matrices --> 100% complete")
        sys.stdout.flush()

        if l2_norm:
            transition = normalize(X=transition,axis=1,norm='l2')
        return transition















if __name__ == '__main__':
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    Trip.setDao(dao)
    #Trip.createAllTrips(save_pickle=True)
    Trip.getTripsPickle()
    #Trip.mapTripToRoad()
    #tmp = 0
