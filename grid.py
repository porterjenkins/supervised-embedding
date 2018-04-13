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
        self.road_list = None


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
        roads = cls.dao.getNodeCoordinates()
        cls.lat_range_city = (roads.latitude.min(),roads.latitude.max() + .00001)
        cls.lon_range_city = (roads.longitude.min(), roads.longitude.max() + .00001)


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
                cls.cell_coord_dict[(cell.lat_range,cell.lon_range)] = cell.id
                id_cnt += 1


    @classmethod
    def mapRoadsToCell(cls):
        print("Mapping roads to cells...")
        if not RoadNode.all_roads.keys():
            RoadNode.init(dao = cls.dao)

        cells_w_road = 0
        for cell in cls.cell_id_dict.values():
            cell.getRoadsInCell()
            if len(cell.road_list) > 0:
                cells_w_road +=1

        print("{} cells of {} contain roads".format(cells_w_road,len(cls.cell_id_dict)))

    @classmethod
    def init(cls,dao):
        GridCell.setDao(dao)
        GridCell.initAllCells(n_grids=20)
        GridCell.mapRoadsToCell()


if __name__ == '__main__':
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    GridCell.setDao(dao)
