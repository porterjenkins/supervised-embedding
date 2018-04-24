from database_access import DatabaseAccess
import numpy as np
from road import RoadNode



class GridCell:

    dao = None
    n_grids = None
    lat_range_city = None
    lon_range_city = None
    lat_cut_points = None
    lon_cut_points = None
    cell_coord_dict = {}
    cell_id_dict = {}


    def __init__(self,id,lat_range=None,lon_range=None):
        self.id = id
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.road_list = list()


    def roadInCell(self,road):
        if road.coordinates[0] >= self.lat_range[0] and road.coordinates[0] < self.lat_range[1]:
            lat_in_range = True
        else:
            lat_in_range = False

        if road.coordinates[1] >= self.lon_range[0] and road.coordinates[1] < self.lon_range[1]:
            lon_in_range = True
        else:
            lon_in_range = False

        return lat_in_range and lon_in_range



    def getRoadsInCell(self):
        road_list = []
        for road_id in RoadNode.all_roads.keys():
            road = RoadNode.all_roads[road_id]
            if self.roadInCell(road):
                road_list.append(road)
                road.updateGridCell(self.id)

        self.road_list = road_list

    @classmethod
    def setDao(cls,dao):
        cls.dao = dao

    @classmethod
    def getCityWindow(cls):

        if cls.dao.lat_range is None and cls.dao.lon_range is None:
            roads = cls.dao.getNodeCoordinates()
            cls.lat_range_city = (roads.latitude.min(),roads.latitude.max() + .00001)
            cls.lon_range_city = (roads.longitude.min(), roads.longitude.max() + .00001)
        else:
            cls.lat_range_city = cls.dao.lat_range
            cls.lon_range_city = cls.dao.lon_range


    @classmethod
    def initAllCells(cls,n_grids):
        cls.n_grids = n_grids
        cls.getCityWindow()
        cls.lat_cut_points = np.linspace(start=cls.lat_range_city[0],stop=cls.lat_range_city[1],num=cls.n_grids+1)
        cls.lon_cut_points = np.linspace(start=cls.lon_range_city[0], stop=cls.lon_range_city[1], num=cls.n_grids+1)

        id_cnt = 0
        for i in range(cls.lat_cut_points.shape[0]-1):
            for j in range(cls.lon_cut_points.shape[0]-1):

                cell = GridCell(id = id_cnt,
                                lat_range=(cls.lat_cut_points[i],cls.lat_cut_points[i+1]),
                                lon_range=(cls.lon_cut_points[j],cls.lon_cut_points[j+1]))

                cls.cell_id_dict[id_cnt] = cell
                #cls.cell_coord_dict[(cell.lat_range,cell.lon_range)] = cell.id
                cls.cell_coord_dict[(i,j)] = cell.id
                id_cnt += 1


    @classmethod
    def mapRoadsToCell(cls):
        print("Mapping roads to cells...")
        if not RoadNode.all_roads.keys():
            RoadNode.init(dao = cls.dao)


        lat_dist = cls.lat_cut_points[1] - cls.lat_cut_points[0]
        lon_dist = cls.lon_cut_points[1] - cls.lon_cut_points[0]


        for id, road in RoadNode.all_roads.items():
            # search latitude
            lat_key = int((road.coordinates[0] - cls.lat_cut_points[0]) // lat_dist)

            if lat_key >= len(cls.lat_cut_points):
                lat_key = None

            # search longitude
            lon_key = int((road.coordinates[1] - cls.lon_cut_points[0]) // lon_dist)
            if lon_key >= len(cls.lon_cut_points):
                lon_key = None

            if lat_key is not None and lon_key is not None:
                cell_id = GridCell.cell_coord_dict[(lat_key,lon_key)]
                road.updateGridCell(cell_id)

                # append road ID to cell's road_list attribute
                cls.cell_id_dict[cell_id].road_list.append(road)



    @classmethod
    def init(cls,dao):
        GridCell.setDao(dao)
        GridCell.initAllCells(n_grids=5)
        GridCell.mapRoadsToCell()


if __name__ == '__main__':
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    GridCell.setDao(dao)
