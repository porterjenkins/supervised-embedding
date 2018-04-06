import numpy as np
import pandas as pd
from database_access import DatabaseAccess


class RoadNode:

    # Class attributes
    all_roads = None
    dao = None

    def __init__(self,node_id,coordinates,adjacent_roads):
        self.node_id = node_id
        self.coordinates = coordinates
        self.adjacent_roads = adjacent_roads
        self.grid_cell = None

    def __str__(self):
        return self.node_id

    def updateAdjacencyList(self,new_list):
        self.adjacent_roads = new_list

    def updateGridCell(self,grid_cell):
        self.grid_cell = grid_cell

    @classmethod
    def setDao(cls,dao):
        cls.dao = dao

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

        all_roads = {}
        for node in nodes.itertuples():
            edge_list = edges[edges.node == node.node]
            adjacent_list = []
            for row in edge_list.itertuples():
                adjacent_list.append(row.adj_node)
            road_i = RoadNode(node_id=node.node,
                          coordinates=[node.latitude,node.longitude],
                          adjacent_roads = adjacent_list)
            all_roads[node.node] = road_i

        all_roads = cls.updateAllAdjacencies(all_roads)
        cls.all_roads = all_roads
        print("{} roads found".format(len(cls.all_roads)))






if __name__ == '__main__':
    dao = DatabaseAccess(city='jinan', data_dir="/Volumes/Porter's Data/penn-state/data-sets/")
    RoadNode.setDao(dao)
    RoadNode.createRoads(road_dir="/Volumes/Porter's Data/penn-state/data-sets/jinan/road-network")