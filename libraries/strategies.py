import numpy as np 
import itertools as it

from queue import LifoQueue
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Callable

Cell = Tuple[int, int]
Neighbor = Tuple[int, Cell]
AdjacencyList = Dict[Cell, Dict[Neighbor, int]]

def reverse_direction(direction:int) -> int:
    """
        This function reverse the input direction : up <=> down; left <=> right 

        :param int direction : the input direction   
        :return: the reversed direction 
    """

    mapper = {
        0:1,  # left to right  
        1:0,  # right to left 
        2:3,  # up to down 
        3:2   # down to up 
    } 
    return mapper[direction]

def heuristic_distance(source_pos:Cell, target_pos:Cell) -> float:
    """
        This function computes the distance between source_pos and target_pos 
        It's based on euclidean_distance formula

        :param source_pos : source position 
        :param target_pos : target position
        
        :return: euclidean_distance between source and target
    """

    source_pos = np.asarray(source_pos)
    target_pos = np.asarray(target_pos)
    return np.sqrt( np.sum( (target_pos - source_pos) ** 2 + 1e-8 ) )

def create_position_mapper(step:int=1) -> List[Callable[[Cell], Tuple[int, Cell]] ]:
    """
        This function create an array of functions 
        Those functions will be used to move over 4 directions(left, right, up, down) 

        :param step: the neighborhood step
        
        :return: array of callables(move pos to neighbor pos) 
    """
    
    return [
        lambda pos: (0, (pos[0], pos[1]-step)),  # left  
        lambda pos: (1, (pos[0], pos[1]+step)),  # right 
        lambda pos: (2, (pos[0]-step, pos[1])),  # up
        lambda pos: (3, (pos[0]+step, pos[1]))   # down 
    ]

def remove_invalid_neighbors(neighbors:List[Tuple[int, Cell]], nb_rows:int, nb_columns:int) -> List[Tuple[int, Cell]]:
    """
        This function will remove invalid neighbors(left, right, up, down)

        :param neighbors: array of neighbors (left, right, up, down)
        :param nb_rows : height of the maze
        :param nb_columns : width of the maze
   
        :return array of valid neighbors
    """
    accumulator = []
    for direction, (row_pos, column_pos) in neighbors:
        row_condition = row_pos >= 0 and row_pos < nb_rows
        column_condition = column_pos >= 0 and column_pos < nb_columns
        if row_condition and column_condition:  # valid position 
            accumulator.append((direction, (row_pos, column_pos)))
    return accumulator

def remove_visited_neighbors(visited_pos:np.ndarray, neighbors:List[Tuple[int, Cell]]) -> List[Tuple[int, Cell]]:
    """
        This function will remove invalid neighbors(left, right, up, down)

        :param visited_pos: binary_matrix(0 or 1)
        :param neighbors_pos: array of neighbors
   
        :return array of unvisited_neighbors 
    """
    accumulator = []
    for direction, position in neighbors:
        if visited_pos[position] != 1:  # not visited neighbor
            accumulator.append((direction, position))
    return accumulator

def backtracking_maze_generator(nb_rows:int, nb_columns:int, seed:int=None, initial_pos:Cell=(0, 0)) -> AdjacencyList:
    """
        This function allows to create a solvable maze
        It's a modified version of recursive_backtracking with a fast convergence 
        Once the maze was built, this function will add a random noise to make it more sparse
        
        :param nb_rows : height of the maze
        :param nb_columns : width of the maze
        :param seed : initial seed of the generator
        :param initial_pos : source_position

        :return an adjacency_list representing the graph 
    """
    if seed is not None:
        np.random.seed(seed)  # fix the generator

    position_mapper = create_position_mapper(step=1)
    visited_pos = np.zeros((nb_rows, nb_columns), dtype=np.uint8)  # copy to avoid side effect 
    
    map_pos2neighbors = {}  # adjacency list for graph representation : to save space  
    positions = list(it.product(range(nb_rows), range(nb_columns)))
    for pos in positions:
        neighbors = [ mapper(pos) for mapper in position_mapper ]
        neighbors = remove_invalid_neighbors(neighbors, nb_rows, nb_columns)
        map_pos2neighbors[pos] = {}
        for ngh in neighbors:
            map_pos2neighbors[pos][ngh] = np.inf # wall 
    # end loop over positions 

    lifo_queue = LifoQueue()  
    lifo_queue.put(initial_pos)
    visited_pos[initial_pos] = 1  # mark first cell as visited 

    while not lifo_queue.empty():
        current_pos = lifo_queue.get()
        neighbors = list(map_pos2neighbors[current_pos].keys())
        unvisited_neighbors = remove_visited_neighbors(visited_pos, neighbors)
        if len(unvisited_neighbors) > 0:
            # simulate recursive call 
            lifo_queue.put(current_pos)  
            # choose one of the unvisted neighbors
            chosen_index = np.random.randint(0, len(unvisited_neighbors))
            chosen_neighbor = unvisited_neighbors[chosen_index]
            chosen_neighbor_direction, chosen_neighbor_position = chosen_neighbor  # unpack neighbor 
            # remove wall between current_pos et neighbors
            current_pos_direction = reverse_direction(chosen_neighbor_direction)
            map_pos2neighbors[current_pos][chosen_neighbor] = 1  # one step to go from current_pos to neighbor
            map_pos2neighbors[chosen_neighbor_position][(current_pos_direction, current_pos)] = 1 
            # mark the chosen neighbor as visisted 
            visited_pos[chosen_neighbor_position] = 1
            # push the chosen neighbor to the stack 
            lifo_queue.put(chosen_neighbor_position)
    # end loop over lifo_queue 
    return map_pos2neighbors

def heuristic_search(map_pos2neighbors:AdjacencyList, initial_pos:Cell, final_pos:Cell) -> Dict[Cell, Cell]:
    """
        This function will search a path between a start node and end node  
        It's based on A*(graph traversal algorithm)
        
        :param map_pos2neighbors: adjacency list of the graph  
        :param initial_pos : source position
        :param final_pos  : target position

        :return hashmap cell => predecessor
    """
    
    positions = map_pos2neighbors.keys()
    row_positions, column_positions = list(zip(*positions))  # unpack [(i,j), ..., (m,n)] => ([i, ..., m], [j, ..., n])
    nb_rows = max(row_positions) + 1
    nb_columns = max(column_positions) + 1

    # initialize predecessors_tracker and visited_pos 
    predecessors_tracker:Dict[int, int] = {initial_pos: -1}
    visited_pos = np.zeros((nb_rows, nb_columns), dtype=np.uint8)  # track cell status(visited or clear) 

    # initialize min_heap and distance_map
    min_heap_scores = []
    map_pos2distances = {}
    
    for pos in positions:
        map_pos2distances[pos] = np.inf 
    
    map_pos2distances[initial_pos] = 0 
    min_heap_scores = [(0, initial_pos)]

    keep_loop = True 
    while keep_loop:
        _, current_pos = heappop(min_heap_scores)  # get the pos with smallest priority : O(1)
        visited_pos[current_pos] = 1 # mark it as visited 

        if current_pos == final_pos:
            keep_loop = False  # a path was found => quit the loop  
        else:
            # go to all 4 directions [left, right, up, down]
            neighbors = list(map_pos2neighbors[current_pos].keys())
            # check if some neighbors are invalid
            remainder_neighbors = remove_visited_neighbors(visited_pos, neighbors)
            # loop over valid neighbors and put them into the fifo_queue
            for neighbor in remainder_neighbors:
                # neighbor is a Tuple[direction, position]
                wall = map_pos2neighbors[current_pos][neighbor]  # infinity if wall else 1 
                updated_distance = map_pos2distances[current_pos] + wall  
                if updated_distance < map_pos2distances[neighbor[1]]:
                    # insert neighbor into the min_heap : O(n*log(n))
                    priority = updated_distance + heuristic_distance(neighbor[1], final_pos) 
                    heappush(min_heap_scores, (priority, neighbor[1]))
                    map_pos2distances[neighbor[1]] = updated_distance 
                    predecessors_tracker[neighbor[1]] = current_pos
            keep_loop = len(min_heap_scores) > 0  # if no neighbors was found, quit the loop : no solution 
    # end loop pathfinding...!

    return predecessors_tracker
        
def build_path(predecessors_tracker:Dict[Tuple[int, int], Tuple[int, int]], initial_pos:Tuple[int, int], final_pos:Tuple[int, int]) -> List[Tuple[int, int]]:
    """
        This function will build the forward path(from start to end) 
        It's based on a hashmap(node=>predecessor) to build the path from bottom(end) to top(start) 

        :param predecessors_tracker : hashmap node => predecessor
        :param initial_pos : source position
        :param final_pos : target position

        :return array of nodes(cell)
    """
    
    solution = []
    if final_pos in predecessors_tracker:
        solution.append(final_pos)
        while solution[-1] != initial_pos:
            predecessor = predecessors_tracker[solution[-1]]
            solution.append(predecessor)
    return solution[-1::-1]  # reverse path_ => from initial_pos to final_pos 

def display_maze(map_pos2neighbors:AdjacencyList, nb_rows:int, nb_columns:int, cell_size:int=3, map_pos2status:Dict[Cell, int]=None) -> str:
    """
        This function will display the maze 

        :param map_pos2neighbors 
        :param nb_rows : height of the maze
        :param nb_columns : width of the maze
        :param cell_size : number of '-' per cell
        :param map_pos2status : status of all nodes(0 => opened_cell, 1 => belongs to path)

        :return string version of the maze 
    """
    assert cell_size % 2 != 0  # cell_size should be an odd number 
    accumulator = []
    top_border = ['-' * cell_size] * nb_columns
    accumulator.append('+' + '+'.join(top_border) + '+')
    for i in range(nb_rows):
        top_acc = []
        right_acc = []
        for j in range(nb_columns):
            pos = (i, j)
            neighbors = list(map_pos2neighbors[pos].keys())
            right_neighbors = [ (n_dir, n_pos) for n_dir, n_pos in neighbors if n_dir == 1]
            cell_data = ' ' * cell_size
            if map_pos2status is not None:
                cell_status = map_pos2status.get(pos, None)
                if cell_status is not None and cell_status == 1: 
                    cell_data = list(cell_data)
                    cell_item = 'S' if pos == (0, 0) else 'E' if pos == (nb_rows - 1, nb_columns - 1) else '#'
                    cell_data[cell_size // 2] = cell_item
                    cell_data = ''.join(cell_data)

            right_acc.append(cell_data)
            if len(right_neighbors) == 1:
                if map_pos2neighbors[pos][right_neighbors[0]] == 1:
                    right_acc.append(' ')
                else: 
                    right_acc.append('|')
            
            top_neighbors = [ (n_dir, n_pos) for n_dir, n_pos in neighbors if n_dir == 2]
            if len(top_neighbors) == 1:
                if map_pos2neighbors[pos][top_neighbors[0]] == 1:
                    top_acc.append(' ' * cell_size)
                else: 
                    top_acc.append('-' * cell_size)
        
        if len(top_acc) > 0:
            accumulator.append('+' + '+'.join(top_acc) + '+')
        accumulator.append('|' + ''.join(right_acc) + '|')


    bottom_border = ['-' * cell_size] * nb_columns
    accumulator.append('+' + '+'.join(bottom_border) + '+')

    return '\n'.join(accumulator)

def build_maze_from_string(string_maze:str) -> Tuple[AdjacencyList, int, int, int]:
    """
        This functions will build the adjacency_list from input string maze 
        :param string_maze : string version of the maze
        
        :return map_pos2neighbors: adjacency list
        :return nb_rows: height of the maze
        :return nb_columns: width of the maze 
        :return cell_size: number of '-' per cell  
    """
    position_mapper = create_position_mapper(step=1)
    
    splited_maze = string_maze.split('\n') 
    splited_maze = [ row[1:-1] for row in splited_maze ]  # ignore left_right outer border 
    top_row = splited_maze[0]
    
    nb_rows = len(splited_maze) // 2 
    nb_columns = len(top_row.split('+'))
    cell_size = len(top_row.split('+')[0])

    map_pos2neighbors = {}  # adjacency list for graph representation : to save space  
    positions = list(it.product(range(nb_rows), range(nb_columns)))
    for pos in positions:
        neighbors = [ mapper(pos) for mapper in position_mapper ]
        neighbors = remove_invalid_neighbors(neighbors, nb_rows, nb_columns)
        map_pos2neighbors[pos] = {}
        for ngh in neighbors:
            map_pos2neighbors[pos][ngh] = np.inf # wall 
    # end loop over positions 

    for pos in positions:
        neighbors = [ mapper(pos) for mapper in position_mapper ]
        neighbors = remove_invalid_neighbors(neighbors, nb_rows, nb_columns)
        row_pos, column_pos = pos 
        maze_row_index = 2 * row_pos + 1
        for direction, neighbor_pos in neighbors:
            direction_ = reverse_direction(direction)
            if direction == 1:  # right 
                current_row_ = splited_maze[maze_row_index]
                if current_row_[(column_pos + 1) * cell_size + column_pos] == ' ':
                    map_pos2neighbors[pos][(direction, neighbor_pos)] = 1       # no wall 
                    map_pos2neighbors[neighbor_pos][(direction_, pos)] = 1
            if direction == 2:  # up 
                top_row_ = splited_maze[maze_row_index - 1].split('+')
                if top_row_[column_pos] == ' ' * cell_size:
                    map_pos2neighbors[pos][(direction, neighbor_pos)] = 1       # wall 
                    map_pos2neighbors[neighbor_pos][(direction_, pos)] = 1     
        # end loop over neighbors for current pos 
    # end loop over positions 

    return map_pos2neighbors, nb_rows, nb_columns, cell_size

