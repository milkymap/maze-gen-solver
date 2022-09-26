import click 

from libraries.log import  logger 
from libraries.strategies import * 

CELL_SIZE = 3

@click.group(chain=True, invoke_without_command=True)
@click.pass_context
def router(ctx):
    ctx.ensure_object(dict)
    subcommad = ctx.invoked_subcommand 
    if subcommad is None:
        logger.warning('use --help option')

@router.command()
@click.option('--width', help='maze width', type=int)
@click.option('--height', help='maze height', type=int)
@click.option('--seed', help='generator seed', type=int, default=None)
def create(width, height, seed):
    maze = backtracking_maze_generator(height, width, seed, (0, 0))
    map_pos2status = {
        (0, 0): 1, 
        (height - 1, width - 1): 1
    }
    str_maze = display_maze(maze, height, width, CELL_SIZE, map_pos2status)    
    print(str_maze)

@router.command()
@click.option(
    '--string_maze', 
    help='output of maze', 
    type=str, 
    callback=lambda ctx,prm,val:click.get_text_stream('stdin').read().strip()
)
def solve(string_maze):
    map_pos2neighbors, nb_rows, nb_columns, cell_size = build_maze_from_string(string_maze)
    
    solution = heuristic_search(map_pos2neighbors, (0, 0), (nb_rows - 1, nb_columns - 1))
    solution = build_path(solution, (0, 0), (nb_rows - 1, nb_columns - 1))

    map_pos2status = {}
    for pos in map_pos2neighbors:
        map_pos2status[pos] = 0
    
    if len(solution) > 0:
        for pos in solution:
            map_pos2status[pos] = 1  # valid path 
        str_maze = display_maze(map_pos2neighbors, nb_rows, nb_columns, cell_size, map_pos2status)
        print(str_maze)
    else:
        logger.debug('The given maze is not solvable')
    
if __name__ == '__main__':
    router(obj={})