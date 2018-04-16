from database_access import DatabaseAccess
import pandas as pd
from pandas import DataFrame, Series
from road import RoadNode
from scipy.spatial.distance import euclidean
import numpy as np
import sys
import pickle
from collections import OrderedDict
import warnings


class TrafficCam:

    # Init class attributes
    dao = None
    all_cams = dict()
    n_date_times = 0

    def __init__(self,id,intersection_name,deployed_date,coordinates,direction,closest_road=None):
        self.id = id
        self.intersection_name = intersection_name
        self.deployed_date = deployed_date
        self.coordinates = coordinates
        self.direction = direction
        self.closest_road = closest_road
        self.traffic_data = pd.DataFrame()
        self.volume = OrderedDict()
        self.n_traffic_dates = None



    def updateClosestRoad(self,road):
        self.closest_road = road

    def updateTrafficData(self,df):
        self.traffic_data = pd.concat((self.traffic_data,df))
        #self.traffic_data.reset_index(inplace=True)

    def updateVolume(self,time_stamp,cnt):
        self.volume[time_stamp] = self.volume.get(time_stamp,0) + cnt

    @classmethod
    def setDao(cls,dao):
        cls.dao = dao

    @classmethod
    def allCamsToPickle(cls):
        fname = cls.dao.cam_dir + "/pickle/all_cams.p"
        pickle.dump(cls.all_cams, open(fname, 'wb'))

    @classmethod
    def getCamsPickle(cls):
        print("Retrieving cameras from pickle...")
        fname = cls.dao.cam_dir + "/pickle/all_cams.p"
        cls.all_cams = pickle.load(open(fname,'rb' ))
        print("{} cameras found".format(len(cls.all_cams)))

    @classmethod
    def getCameraTrafficData(cls,n_records_file = None,save_pickle=False):
        print("Getting camera traffic data...")
        dates = cls.dao.getCamTrafficFiles()
        warnings.filterwarnings('error')
        all_cols = ["license_plate","license plate_category","veh_brand","veh_color","license_plate_color",'camera_id',
                'time','speed','lane','veh_direction','driving_state','location_id','cam_direction']
        use_cols = ['camera_id','license_plate','time','speed','cam_direction']
        cnt = 0


        for date in dates:
            cnt += 1
            date_data = pd.read_csv(cls.dao.cam_dir + "/traffic/" + date,
                                    names=all_cols,
                                    sep=';',
                                    nrows=n_records_file,
                                    usecols=use_cols)

            # Group into 1 hour intervals
            #date_slice = date_data['time'].str.slice(start=0,stop=19)
            #date_data['time'] = pd.to_datetime(date_slice)
            #date_data.set_index(pd.DatetimeIndex(date_data['time']),inplace=True)
            #hourly_volume = date_data.groupby(pd.TimeGrouper(freq='60Min')).size()


            cams_in_date_data = np.unique(date_data.camera_id)
            for cam in cls.all_cams.values():
                if cam.id in cams_in_date_data:
                    cam_dta = date_data[date_data.camera_id == cam.id]
                    cam.updateTrafficData(cam_dta)
                    cam.updateVolume(date.split(".")[0],cam_dta.shape[0])
                else:
                    cam.updateVolume(date.split(".")[0], 0)



            #for cam_id in cams_in_date_data:
            #    cam = cls.all_cams.get(cam_id)
            #    if cam:
            #        cam_dta = date_data[date_data.camera_id == cam.id]
            #        cam.updateTrafficData(cam_dta)
            #        cam.updateVolume(date.split(".")[0],cam_dta.shape[0])
            #    else:


            progress = round((cnt / len(dates))*100,2)

            sys.stdout.write("\r Progress: {}% of dates complete".format(progress))
            sys.stdout.flush()

        if save_pickle:
            cls.allCamsToPickle()



    @classmethod
    def createAllCams(cls):
        print("Getting camera metadata from file...")
        fname = cls.dao.cam_dir + "/summaries/kkdw_jinan_all.csv"
        all_cams_df = pd.read_csv(fname,header=None,index_col=0)
        all_cams_trim = all_cams_df.iloc[:,0:10]
        all_cams_trim.columns = ["cameraID", "intersectionName", "cameraProviderName",
                            "YearMonth", "AID", "Longitude", "Latitude", "BID",
                            "intersectionName", "direction" ]


        for row in all_cams_trim.itertuples():

            # trim camera ids to match traffic data
            if len(row.cameraID.split()) == 2:
                cam_id = int(row.cameraID.split()[1])
            else:
                cam_id = int(row.cameraID)

            camera = TrafficCam(id=cam_id,
                                intersection_name=row.intersectionName,
                                deployed_date=row.YearMonth,
                                coordinates=[row.Latitude,row.Longitude],
                                direction=row.direction)

            cls.all_cams[cam_id] = camera
        print("{} cameras found".format(len(cls.all_cams)))


    @classmethod
    def mapCamToRoads(cls,save_pickle=False):
        print("\nMapping cameras to roads...")
        if not RoadNode.all_roads.keys():
            RoadNode.init(dao = cls.dao)
        n_roads = len(RoadNode.all_roads)
        n_camera = len(TrafficCam.all_cams)

        cnt = 0
        for cam_id in cls.all_cams.keys():
            camera = cls.all_cams[cam_id]
            dist = np.zeros(shape=n_roads)
            road_ids = np.zeros(shape=n_roads,dtype=np.int32)
            for i,road_id in enumerate(RoadNode.all_roads.keys()):
                cnt += 1
                road_ids[i] = road_id
                dist[i] = euclidean(camera.coordinates,RoadNode.all_roads[road_id].coordinates)

                progress = round((cnt / float(n_roads*n_camera)) * 100, 2)

                sys.stdout.write("\r Camera mapping progress: {}% complete".format(progress))
                sys.stdout.flush()


            closest_road_idx = np.argmin(dist) # find closest road index
            closest_road = RoadNode.all_roads[road_ids[closest_road_idx]] # closest road
            closest_road.addCamera(new_cam=camera) # append current camera to closest road's camera list
            camera.updateClosestRoad(road_ids[closest_road_idx]) # add camera id to cma
        print('')

        if save_pickle:
            cls.allCamsToPickle()
            RoadNode.allRoadsToPickle()

    @classmethod
    def init(cls,dao,read_pickle):
        TrafficCam.setDao(dao=dao)
        if read_pickle:
            TrafficCam.getCamsPickle()
            # update roads after mapping cameras and roads
            #RoadNode.getRoadsPickle()
        else:
            TrafficCam.createAllCams()
            TrafficCam.getCameraTrafficData(save_pickle=True)

        TrafficCam.mapCamToRoads()


if __name__ == '__main__':
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    TrafficCam.setDao(dao=dao)
    TrafficCam.createAllCams()
    TrafficCam.getCameraTrafficData(save_pickle=True)
    #TrafficCam.mapCamToRoads(save_pickle=True)


