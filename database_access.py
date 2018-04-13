import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class DatabaseAccess:

    def __init__(self,city,data_dir=None):
        self.city = city.lower()
        if data_dir is None:
            self.data_dir = os.getcwd() +"/data/"+ self.city
        else:
            self.data_dir = data_dir + self.city

        self.traj_dir = self.data_dir + "/trajectory"
        self.cam_dir = self.data_dir + "/street-cam"
        self.road_dir = self.data_dir + "/road-network"



    def getTrajFiles(self):
        file_dict = {}
        dates = os.listdir(self.traj_dir)
        if '.DS_Store' in dates:
            dates.remove(".DS_Store")
        if 'summaries' in dates:
            dates.remove('summaries')
        if 'pickle' in dates:
            dates.remove('pickle')



        for date in dates:
            file_dict[date] = os.listdir(self.traj_dir + "/" + date)
        return file_dict

    def getCamTrafficFiles(self):
        dates = os.listdir(self.cam_dir + "/traffic")
        return dates


    def readGpsData(self,date,file):
        fname = self.traj_dir + "/" + date + "/" + file
        dta =  pd.read_csv(fname,engine='c')
        if dta.empty:
            return None
        else:
            dta['time_diff'] = dta['timestamp'].diff(periods=1).fillna(0)
            return dta



    def readObdData(self,date,file):
        header = ["totalmileage","totalfuel","enginespeed","OBDspeed","devicesn","mileage","timestamp","raw"]
        fname = self.data_dir + "/" + date + "/" + file
        dta = pd.read_csv(fname,
                           sep=',',
                           names=header,
                           usecols=header[:-1])
        dta.drop(labels=0,axis=0,inplace=True)
        if dta.empty:
            return None
        else:
            return dta


    def getEdgeList(self):
        fname = self.road_dir + "/bigedges.txt"
        return pd.read_csv(fname,
                           header=None,
                           names = ['node','adj_node'])

    def getNodeCoordinates(self):
        fname = self.road_dir + "/bignodes.txt"
        return pd.read_csv(fname,
                           header = None,
                           names = ['node','longitude','latitude'])

    def getNodeLanes(self):
        fname = self.road_dir + "/bignode_lanes.txt"
        return pd.read_csv(fname,
                           header=None,
                           index_col = 0,
                           names=['lane_cnt'])





class DatabaseSummary(DatabaseAccess):

    def __init__(self,city,data_type,data_dir):
        super().__init__(city,data_dir)
        self.data_type = data_type
        self.gps_summary = {'latitude':[],
                            'longitude':[],
                            'n_car_trips':0,
                            'n_days':0,
                            'unique_car_gps_ids':set(),
                            'trajectory_record_cnt':0,
                            'odb_record_cnt':0,
                            'n_unique_cars': 0,
                            'n_road_segments': 2238,
                            'n_cameras': 2007,
                            'car_time_per_day':[],
                            'mean_car_time_per_day':0}
        if data_dir is None:
            self.data_dir = os.getcwd() +"/data/"+ self.city+"/"+self.data_type
        else:
            self.data_dir = data_dir + self.city+"/"+self.data_type
        self.data_sources = ['road-network','street-cam','trajectory']


    def updateGpsSummaryStats(self,dta):
        self.gps_summary['latitude'] += list(dta['latitude'].values)
        self.gps_summary['longitude'] += list(dta['longitude'].values)
        self.gps_summary['trajectory_record_cnt'] += len(dta)

        unique_ids_file = dta['devicesn'].unique()
        for id in unique_ids_file:
            self.gps_summary['unique_car_gps_ids'].add(id)
        self.gps_summary['n_unique_cars'] = len(self.gps_summary['unique_car_gps_ids'])

        self.gps_summary['n_car_trips'] += len(dta['time_diff'][dta['time_diff'] >=60])+ 1
        self.gps_summary['car_time_per_day'].append(round((dta.loc[dta.shape[0]-1,'timestamp'] - dta.loc[0,'timestamp'])/(60.0*60.0),2))



    def updateOdbSummaryStats(self,dta):

        self.gps_summary['odb_record_cnt'] += len(dta)
        return None

    def writeSummaries(self,n_random=None):

        keys = list(self.gps_summary.keys())
        # Write lat/long into separate file
        location = pd.DataFrame.from_items(items=[('latitude',self.gps_summary['latitude']),
                                                  ('longitude',self.gps_summary['longitude'])])
        if n_random is not None and n_random < location.shape[0]:
            idx = np.random.permutation(location.index)
            idx_sample = idx[:n_random]
            location = location.ix[idx_sample]

        location.to_csv(self.data_dir+"/summaries/trajectories_all.csv",index=False)
        self.gps_summary['mean_car_time_per_day'] = np.mean(self.gps_summary['car_time_per_day'])

        keys.remove('latitude')
        keys.remove('longitude')
        keys.remove('unique_car_gps_ids')
        keys.remove('car_time_per_day')
        f = open(self.data_dir+"/summaries/summary_stats.txt",'w')
        for key in keys:
            f.write("{}: {}\n".format(key,self.gps_summary[key]))

        f.close()



    def getSummaryStats(self,write=False,n_random=None,n_days=None):
        files = self.getTrajFiles()
        dates = list(files.keys())
        self.gps_summary['n_days'] = len(dates)
        if n_days is not None:
            dates = dates[:n_days]
        for date in dates:
            print(date)
            for f in files[date]:
                if 'gps' in f:
                    dta = self.readGpsData(date,f)
                    #
                    if dta is None:
                        pass
                    else:
                        # update gps summary stats
                        self.updateGpsSummaryStats(dta)
                elif 'obd':
                    dta = self.readObdData(date,f)
                    if dta is None:
                        pass
                    else:
                        # update odb summary stats
                        self.updateOdbSummaryStats(dta)

        if write:
            self.writeSummaries(n_random)

        return None



class SummaryPlots:

    def __init__(self,city,data_dir=None):
        self.city = city.lower()
        if data_dir is None:
            self.data_dir = os.getcwd() +"/data/"+ self.city
        else:
            self.data_dir = data_dir + self.city


    def plotTrajectoryWithCams(self):

        trajectories = pd.read_csv(self.data_dir+"/trajectory/summaries/trajectories_all.csv",index_col=None)
        cameras = pd.read_csv(self.data_dir+"/street-cam/summaries/kkdw_jinan_all.csv")

        plt.scatter(y=trajectories['latitude'],
                    x=trajectories['longitude'],
                    s =1.5,
                    c='blue',
                    alpha=.33)
        plt.scatter(y=cameras.iloc[:,7],
                    x=cameras.iloc[:,6],
                    s=1.5,
                    c='red',
                    alpha=.33)

        plt.show()







if __name__ == '__main__':
    dao = DatabaseSummary(city='jinan',data_type='trajectory',data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    dao.getSummaryStats(write=True,n_random=1000000)

    map = SummaryPlots(city='jinan',data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    map.plotTrajectoryWithCams()

