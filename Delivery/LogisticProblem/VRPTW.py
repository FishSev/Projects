# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:46:27 2020.

@authors: Dmitrij
          FishSev
"""

from __future__ import print_function
# import os
import copy
import random
import math
import numpy
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.path import Path
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from time import perf_counter

# Global variables
# OR-Tools Model Vars
time_limit = 10                # time limit to find a solution in seconds
penalty = 1000                 # penalty for dropping a node. as it hier it is easier to drop, so make it larger than traveling time between the nodes # noqa
pickups_deliveries = [[2, 4]]  # obligatory points, can't drop this condition (at least for the moment) # noqa

# Nodes params
min_range = 0                   # min coordinates values on plot
max_range = 50                  # max coordinates values on plot
minutes_in_hour = 60
warehouses_number = 2           # number of warehouses
shops_number = 2                # number of shops
shop_min_time = 9*minutes_in_hour   # shop start working time
shop_max_time = 18*minutes_in_hour  # shop end working time
customers_number = 10            # number of customers
max_load_time = 40
max_unload_time = 20            # maximum unload time in minutes
nodes_max_demand = 10           # nodes maximum demand
node_size = max_range           # node size on plot
min_time = 0                  # min time window boundary
max_time = 24*minutes_in_hour                 # max time window boundary
distance_to_time_ratio = 25/minutes_in_hour   # ratio to calculate travel times

# Vehicle params
vehicles_number = 2              # number of vehicles
standart_capacity = 30         # vehicle standart capacity
waiting_time = minutes_in_hour*24           # allow waiting time, between nodes (if small may not find a solution) # noqa
max_vehicle_work_time = 24*minutes_in_hour  # maximum time per vehicle

# Creating pseudo random values
# Fixing random state for reproducibility
random.seed(1)  # ok
output = []


class tms_Node():
    """Сlass operates with Nodes parameters."""

    def __init__(self,
                 coordinates=None,
                 demand=None,
                 node_type='Customer',
                 time_window=None,
                 initial_index=None,
                 unload_time=None
                 ):

        # Set initial_index
        self.initial_index = None

        # Set coordinates
        if (coordinates is not None):
            self.coordinates = coordinates
        else:
            self.coordinates = (random.randint(min_range, max_range),
                                random.randint(min_range, max_range))

        # Set demand
        if (demand is not None):
            self.demand = demand
        else:
            self.demand = random.randint(1, nodes_max_demand)
        self.node_type = node_type

        # Set time windows
        if (time_window is not None):
            self.time_window = time_window
        else:
            if (self.node_type == 'Warehouse'):
                self.time_window = (min_time, max_time)
            else:
                if (self.node_type == 'Shop'):
                    self.time_window = (shop_min_time, shop_max_time)
                else:
                    time_from = random.randint(min_time, max_time//2)
                    time_to = random.randint(time_from + 1, max_time)
                    self.time_window = (time_from, time_to)

        # Set unload time
        if (unload_time is not None):
            self.unload_time = unload_time
        else:
            if self.node_type == 'Warehouse':  # For Shop-node unload means time for load the product # noqa
                self.unload_time = 0
            else:
                self.unload_time = random.randint(1, max_unload_time)


def draw_nodes(tms_nodes, ax):
    """Draw tms nodes."""
    x = []
    y = []
    colors = []
    areas = []
    capacities = []
    for node in tms_nodes:
        # Coordinates
        x.append(node.coordinates[0])
        y.append(node.coordinates[1])

        # Capacities
        capacities.append(node.demand)

        # Types
        if (node.node_type == 'Warehouse'):
            colors.append('black')
            areas.append(node_size*2)
        else:
            if (node.node_type == 'Shop'):
                colors.append('green')
                areas.append(node_size)
            else:
                colors.append('blue')
                areas.append(node_size/2)

    ax.set_xlim(min_range, max_range)
    ax.set_ylim(min_range, max_range)
    ax.scatter(x, y, s=areas, c=colors, alpha=0.5)


def calculate_distance(tms_Node_from, tms_Node_to):
    """Return euclidian distance between given nodes."""
    x_diff = abs(tms_Node_from.coordinates[0] - tms_Node_to.coordinates[0])
    y_diff = abs(tms_Node_from.coordinates[1] - tms_Node_to.coordinates[1])
    return math.sqrt(pow(x_diff, 2) + pow(y_diff, 2))/distance_to_time_ratio


def generate_distance_matrix(tms_nodes=None):
    """Generate distance matrix from nodes."""
    tms_matrix = []
    if (tms_nodes is not None):
        for i in tms_nodes:
            tms_distance_row = []
            for j in tms_nodes:
                tms_distance_row.append(calculate_distance(i, j))
            tms_matrix.append(tms_distance_row)
    else:
        tms_matrix = None
    if (tms_matrix is not None):
        return tms_matrix


class tms_Vehicles():
    """Сlass operates with Vehicles parameters."""

    def __init__(self,
                 vehicle_type='Common',
                 capacity=standart_capacity,
                 time_windows=(min_time, max_time)):
        self.vehicle_type = vehicle_type
        self.capacity = capacity
        self.time_windows = time_windows


""" -----Time to clock-------"""


def TTC(time):
    """Return time in hh:mm format."""
    time = str(int(time//60)) + ':' + str(int(round(time % 60)))
    return(time)


def PTM(matrix):
    """Print given matrix."""
    for i in range(numpy.shape(matrix)[0]):
        for j in range(numpy.shape(matrix)[1]):
            print('{}'.format(TTC(matrix[i][j])).center(6, ' '), end=' ')
        print()
    print()


def add_unload_time(data):
    """Add unload time to nodes as an additional time in time_matrix."""
    for from_node in range(len(data['unload_times'])):
        for to_node in range(numpy.shape(data['time_matrix'])[1]):
            if from_node != to_node:
                data['time_matrix'][from_node][to_node] += data['unload_times'][from_node]  # noqa


""" --------OR-Tools--------"""


def create_data_model():
    """Store the data for the problem."""
    # Generating nodes
    tms_nodes = []

    for i in range(warehouses_number):
        tms_nodes.append(tms_Node((random.randint(min_range, max_range),
                                  random.randint(min_range, max_range)),
                                  0, 'Warehouse'))

    for i in range(shops_number):
        tms_nodes.append(tms_Node((random.randint(min_range, max_range),
                                  random.randint(min_range, max_range)),
                                  0, 'Shop'))

    for count in range(customers_number):
        tms_nodes.append(tms_Node())

    time_windows = []
    demands = []
    unload_times = []

    for node in tms_nodes:
        node.initial_index = tms_nodes.index(node)
        time_windows.append(node.time_window)
        demands.append(node.demand)
        unload_times.append(node.unload_time)

    data = {}
    data['tms_nodes'] = tms_nodes
    data['time_matrix'] = generate_distance_matrix(tms_nodes)
    data['time_windows'] = time_windows
    data['demands'] = demands
    data['vehicle_capacities'] = [standart_capacity]*vehicles_number
    data['num_vehicles'] = vehicles_number
    data['unload_times'] = unload_times

    data['starts'] = []
    data['ends'] = []
    data['load_times'] = []
    for count in range(vehicles_number):
        data['starts'].append(random.randint(0, warehouses_number-1))
        data['ends'].append(random.randint(0, warehouses_number-1))
        data['load_times'].append(random.randint(1, max_load_time))
    data['pickups_deliveries'] = pickups_deliveries

    return data


def draw_path(data, manager, routing, solution):
    """Draw the route."""
    global full_path
    full_path = []
    for vehicle_id in range(data['num_vehicles']):
        path = []
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        path.append(manager.IndexToNode(index))
        full_path.append(path)


def print_solution(data, manager, routing, solution):
    """Print the solution in console."""
    tms_nodes = data['tms_nodes']
    time_dimension = routing.GetDimensionOrDie('Time')
    global output

    for vehicle_id in range(data['num_vehicles']):
        path_data = {}
        path_data['vehicle'] = vehicle_id
        path = []
        times = []
        weight = 0

        index = routing.Start(vehicle_id)

        if len(output) < data['num_vehicles']:
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        else:
            plan_output = ''
        counter = 0

        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            path.append(manager.IndexToNode(index))
            weight += tms_nodes[manager.IndexToNode(index)].demand
            times.append((solution.Min(time_var), solution.Max(time_var)))

            plan_output += '{3} -> {0} ({1}-{2}) | '.format(tms_nodes[manager.IndexToNode(index)].initial_index, TTC(solution.Min(time_var)), TTC(solution.Max(time_var)), tms_nodes[path[len(path)-2]].initial_index if (len(path)-2 != -1) else 'start')  # noqa
            index = solution.Value(routing.NextVar(index))
            counter += 1

        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} ({1},{2})\n'.format(tms_nodes[manager.IndexToNode(index)].initial_index, TTC(solution.Min(time_var)), TTC(solution.Max(time_var)))  # noqa

        if len(output) < vehicles_number:
            output.append(plan_output)
        else:
            output[vehicle_id] += plan_output+" "


# Print data in a given form
def data_print(data):
    """Print result."""
    print('\n')
    print('Start ', data['starts'])
    print('End ', data['ends'])
    print('Demands', data['demands'])
    print('Deliver ', data['pickups_deliveries'])
    print('Unload time', data['unload_times'])
    print('Load time for each vehicle', data['load_times'], '\n')
    print
    print('W - Warehouse, C - Customer, S - Shop, D -demand, TW - time window\n')  # noqa

    for node in data['tms_nodes']:
        print(
            '{} {}, type - {}, D = {}, TW - ({}-{})'
            .format(
                    data['tms_nodes'].index(node),
                    node.coordinates,
                    'W' if node.node_type == 'Warehouse' else 'S' if node.node_type == 'Shop' else 'C',  # noqa
                    node.demand,
                    TTC(node.time_window[0]),
                    TTC(node.time_window[1])
                    )
            )
    print('\n')


def delete_matrix_column(mtx: list, index: int) -> list:
    """Delete column in the given matrix."""
    for row in mtx:
        row.pop(index)
    return mtx


def clear_from_empty(data, empty, type):
    """# Clear the data from empty elements."""
    for element in reversed(empty):
        del data['tms_nodes'][element]
        del data['time_windows'][element]
        del data['demands'][element]
        data['time_matrix'].pop(element)
        delete_matrix_column(data['time_matrix'], element)

        if type == 'warehouse':
            for elem in range(len(data['starts'])):
                if data['starts'][elem] > element:
                    data['starts'][elem] -= 1
                if data['ends'][elem] > element:
                    data['ends'][elem] -= 1

        for row in range(len(data['pickups_deliveries'])):
            for column in range(len(data['pickups_deliveries'][0])):
                if data['pickups_deliveries'][row][column] > element:
                    data['pickups_deliveries'][row][column] -= 1

    return(data)


def renew_time(data, manager, routing, solution):
    """# Renew time windows for vehicles."""
    global new_time_windows
    new_time_windows = []

    time_dimension = routing.GetDimensionOrDie('Time')

    for vehicle_id in range(data['num_vehicles']):

        index = routing.Start(vehicle_id)

        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            index = solution.Value(routing.NextVar(index))

        time_var = time_dimension.CumulVar(index)
        new_time_windows.append(solution.Max(time_var))


def main():
    """Solve the VRP with time windows."""
    start_time = perf_counter()

    # Instantiate the data problem.
    data = create_data_model()

    initial_data = copy.deepcopy(data)
    print('Initial problem')
    data_print(initial_data)

    # Add unload time to the nodes
    add_unload_time(data)

    global shops_number
    global warehouses_number

    empty_shops = []
    column1 = [r[0] for r in data['pickups_deliveries']]
    for shop in range(shops_number):
        if (shop+warehouses_number not in column1):
            if data['tms_nodes'][shop+warehouses_number].node_type == "Shop":
                empty_shops.append(shop+warehouses_number)

    data = clear_from_empty(data, empty_shops, 'shop')
    shops_number -= len(empty_shops)

    empty_warehouses = []
    for warehouse in range(warehouses_number):
        if (warehouse not in data['starts'] and warehouse not in data['ends']):
            empty_warehouses.append(warehouse)
    data = clear_from_empty(data, empty_warehouses, 'warehouse')
    warehouses_number -= len(empty_warehouses)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'],
                                           data['starts'],
                                           data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Return the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        waiting_time,           # allow waiting time
        max_vehicle_work_time,  # maximum time per vehicle
        False,                   # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    # Add load times
    for vehicle in range(vehicles_number):
        index = routing.Start(vehicle)
        time_dimension.CumulVar(index).SetRange(data['load_times'][vehicle] +
                                                data['time_windows'][data['starts'][vehicle]][0], max_time)  # noqa

    # Add time window constraints for each location except depots.
    for location_idx, time_window in enumerate(data['time_windows']):
        if (location_idx < warehouses_number):
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
        routing.solver().Add(
            time_dimension.CumulVar(pickup_index) <=
            time_dimension.CumulVar(delivery_index))

        # Add Capacity constraint.
    def demand_callback(from_index):
        """Return the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    capacity = 'Capacity'
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        capacity)

    # Allow to drop nodes.
    for node in range(1, len(data['time_matrix'])):
        if not(manager.NodeToIndex(node) == -1):  # don't know why i need this
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    search_parameters.solution_limit = 1
    search_parameters.time_limit.seconds = time_limit

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
        unsolved_part = False
    else:
        print("No initial solution found")
        unsolved_part = True

    # Solve the whole problem

    previous_clear = []
    while True:

        if (unsolved_part):
            break

        draw_path(data, manager, routing, solution)

        not_visited_nodes = []
        for count in range(len(data['tms_nodes'])):
            not_visited_nodes.append(count)

        for node in not_visited_nodes:
            for _list in full_path:
                for _element in _list:
                    if _element in not_visited_nodes:
                        not_visited_nodes.remove(_element)

        for warehouse in range(warehouses_number):
            not_visited_nodes.append(warehouse)

        for pickups in data['pickups_deliveries']:
            if pickups[1] not in not_visited_nodes:
                data['pickups_deliveries'].remove(pickups)

        empty_shops = []
        column1 = [r[0] for r in data['pickups_deliveries']]
        for shop in range(shops_number):
            if (shop+warehouses_number not in column1):
                if data['tms_nodes'][shop+warehouses_number].node_type == "Shop":  # noqa
                    empty_shops.append(shop+warehouses_number)

        clear_from_empty(data, empty_shops, 'shop')
        shops_number -= len(empty_shops)

        empty_shops.sort()
        for shop in reversed(empty_shops):
            for count in range(len(not_visited_nodes)):
                if not_visited_nodes[count] > shop:
                    not_visited_nodes[count] -= 1

        not_visited_nodes.sort()
        not_visited_nodes.reverse()

        if ((previous_clear == not_visited_nodes) or
                (not_visited_nodes[0] <= warehouses_number)):
            break

        visited_nodes = []
        for count in range(len(data['tms_nodes'])):
            if count not in not_visited_nodes:
                visited_nodes.append(count)

        clear_from_empty(data, visited_nodes, 'customer')

        data['starts'], data['ends'] = data['ends'], data['starts']

        renew_time(data, manager, routing, solution)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                               data['num_vehicles'],
                                               data['starts'],
                                               data['ends'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Time Windows constraint.
        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            waiting_time,           # allow waiting time
            max_vehicle_work_time,  # maximum time per vehicle
            False,                   # Don't force start cumul to zero.
            time)
        time_dimension = routing.GetDimensionOrDie(time)

        # define new start time for each vehicle including load time at depo
        for vehicle in range(vehicles_number):
            index = routing.Start(vehicle)
            time_dimension.CumulVar(index).SetRange(new_time_windows[vehicle] +
                                                    data['load_times'][vehicle], max_time)  # noqa

        # Add time window constraints for each location except depots.
        for location_idx, time_window in enumerate(data['time_windows']):
            if (location_idx < warehouses_number):
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0],
                                                    time_window[1])
        # Add time window constraints for each vehicle start node.
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                    data['time_windows'][0][1])

        # Instantiate route start and end times to produce feasible times.
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i)))

        for request in data['pickups_deliveries']:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(
                    delivery_index))
            routing.solver().Add(
                time_dimension.CumulVar(pickup_index) <=
                time_dimension.CumulVar(delivery_index))

        capacity = 'Capacity'
        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            capacity)

        # Allow to drop nodes.
        for node in range(1, len(data['time_matrix'])):
            if not(manager.NodeToIndex(node) == -1):  # don't know why i need this  # noqa
                routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
        search_parameters.solution_limit = 1
        search_parameters.time_limit.seconds = time_limit

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            print_solution(data, manager, routing, solution)
        else:
            print('========')
            print('Unsolved part of the problem:')
            data_print(data)
            print('========')

            unsolved_part = True

            break

        previous_clear = not_visited_nodes

    if unsolved_part:
        print('See unsolved part of the problem above ^^^\n========\n')

    for element in output:
        print(output[output.index(element)])

    # Print program execution time
    TotalTime = perf_counter() - start_time
    print("Время выполнения", '%.2f' % TotalTime)


if __name__ == '__main__':
    main()
