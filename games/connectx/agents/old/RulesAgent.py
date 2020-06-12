def RulesAgent( observation, configuration, verbose=True ):
    verbose = True
    try:
        from functools import lru_cache
        from itertools import product
        from typing import Union, List, Tuple, FrozenSet, Set
        from collections import defaultdict
        import numpy as np
        import random
        import traceback


        board = np.array(observation.board).reshape(configuration.rows, configuration.columns)
        hero_mark    = observation.mark
        villain_mark = 1 if hero_mark == 2 else 2

        # (1,0)  -> (-1,0)  = down -> up
        # (0,1)  -> (0,-1)  = left -> right
        # (1,1)  -> (-1,-1) = down+left -> up+right
        # (-1,1) -> (1,-1)  = up+left   -> down+right
        all_directions  = frozenset(
            frozenset([ (x,y), (-x,-y) ])
            for x,y in product([-1,0,1],[-1,0,1])
            if not (x == y == 0)
        )
        playable_row = [ np.count_nonzero( board[:,col] == 0 ) - 1 for col in range(configuration.columns)      ]
        played_cols  = [ col for col in range(configuration.columns) if np.count_nonzero( board[:,col] ) != 0   ]
        empty_cols   = [ col for col in range(configuration.columns) if np.count_nonzero( board[:,col] ) == 0   ]
        valid_cols   = { col for col in range(configuration.columns) if board[0,col] == 0 }  # rows are counted from sky = 0
        valid_cells  = { (playable_row[col],col) for col in range(configuration.columns) if playable_row[col] != -1 }
        middle_column  = configuration.columns//2
        is_board_empty = np.count_nonzero(observation.board) == 0

        def get_cell_value(cell: Union[Tuple[int,int], None]) -> Union[int,None]:
            if cell is None: return None
            return board[cell[0],cell[1]]

        # @lru_cache()
        def is_valid_move(col: Union[int,Tuple[int,int]]):
            if isinstance(col, tuple): col = col[1]
            if col is None: return False
            return col in valid_cols

        # def get_playable_row(col: int) -> Union[int,None]:
        #     row = np.count_nonzero( board[:,col] == 0 ) - 1
        #     return row if row >= 0 else None

        def get_cells_by_intersection_score( mark: int, only_playable=True ):
            scores = defaultdict(int)
            lines  = get_lines(mark)
            for line in lines:
                for cell in line:
                    if only_playable and not is_valid_move(cell): continue
                    scores[cell] += get_line_length(line)
            output = sorted([ (score, cell) for score, cell in zip(scores.values(), scores.keys()) ])
            return output

        # @lru_cache()
        def get_lines_by_length(mark: int) -> List[Tuple[int, Tuple[Tuple[int, int]]]]:
            lines  = get_lines(mark)
            output = [ (get_line_length(line), line) for line in lines ]
            output = sorted(output, reverse=True)
            return output

        def get_line_length(line: Tuple[Tuple[int,int]]) -> int:
            values = [ board[cell[0],cell[1]] for cell in line ]
            length = np.count_nonzero(values)
            return length

        # @lru_cache()
        def get_lines(mark: int) -> List[Tuple[Tuple[int,int]]]:
            output = set()
            for coords in product(range(configuration.rows), range(configuration.columns)):
                if get_cell_value(coords) == 0: continue
                for directions in all_directions:
                    line = get_line(coords, directions, mark=mark)
                    if len(line) < configuration.inarow: continue
                    output.add(line)
            return list(output)

        # @lru_cache()
        def get_line(coords: Tuple[int,int], directions: FrozenSet[Tuple[int,int]], mark: int) -> Tuple[Tuple[int,int]]:
            line  = [ coords ]
            for direction in directions:
                count = 0
                cell  = coords
                value = board[cell[0],cell[1]]
                if value != mark: return tuple()

                while count <= configuration.inarow:
                    cell  = next_cell(cell, direction)
                    if cell is None:        break;                                   # Lines terminate at the edge of the board

                    value = board[cell[0],cell[1]]
                    if   value == 0:        count += 1; line.append(cell); continue  # Lines count to inarow in empty space
                    elif value == mark:     count  = 0; line.append(cell); continue  # Lines extend inarow beyond the last disk of same color
                    elif value != mark:     break                                    # lines terminates at opposite token

            line = tuple(sorted(line))
            return line


        # @lru_cache()
        def get_line_edges(line: List[Tuple[int,int]], mark: int) -> List[Tuple[int,int]]:
            values = [ board[cell[0],cell[1]] for cell in line ]
            if mark not in values: return []
            edges  = []
            last_value = None
            last_cell  = None
            for cell, value in zip(line,values):
                if   value == mark and last_value == 0:    edges.append(last_cell)
                elif value == 0    and last_value == mark: edges.append(cell)
                last_value = value
                last_cell  = cell

            return sorted(set(edges))

        # @lru_cache()
        def get_line_zero_groups(line: List[Tuple[int,int]]) -> List[List[Tuple[int,int]]]:
            values = [ board[cell[0],cell[1]] for cell in line ]
            output = []
            buffer = []
            last   = None
            for cell, value in zip(line,values):
                if   value == 0:  buffer.append(cell);   continue
                elif len(buffer): output.append(buffer); buffer = [];
            if len(buffer):       output.append(buffer); buffer = [];
            return output

        # # @lru_cache()
        # def get_line_zero_groups_lengths(line: List[Tuple[int,int]]) -> List[int]:
        #     output = list(map(len, get_line_zero_groups(line)))
        #     return output

        # @lru_cache()
        def get_line_middle(line: List[Tuple[int,int]], mark: int) -> List[Tuple[int,int]]:
            values = [ board[cell[0], cell[1]] for cell in line ]
            middle  = []
            buffer  = []
            last    = None
            middle_started = False
            for index, (cell,value) in enumerate(zip(line,values)):
                if value == 0 and last is mark:      middle_started = True   # exclude left hand zeros
                if value == 0 and middle_started:    buffer.append(cell)
                if value == mark and middle_started: middle += buffer; buffer = []; middle_started = False
                if index == len(line)-1:             buffer = []; break      # exclude right hand zeros
            return sorted(middle)

        # @lru_cache()
        def get_playable_line_edges(line: List[Tuple[int,int]], mark: int) -> List[Tuple[int,int]]:
            edges  = [ cell for cell in get_line_edges(line, mark) ]
            output = [ cell for cell in edges if is_valid_move(cell) ]
            return output

        # # @lru_cache()
        # def get_playable_line_middle(line: List[Tuple[int,int]], mark: int) -> List[Tuple[int,int]]:
        #     output = [ cell for cell in get_line_middle(line, mark) if is_valid_move(cell) ]
        #     return output

        # @lru_cache()
        def next_cell(coords: Tuple[int,int], direction: Tuple[int,int]) -> Union[Tuple[int,int],None]:
            if coords is None or direction is None:                 return None

            output = ( coords[0] + direction[0], coords[1] + direction[1] )
            if output[0] < 0 or output[0] >= configuration.rows:    return None
            if output[1] < 0 or output[1] >= configuration.columns: return None
            return output


        ### Strategies ###


        # best opening strategy is to play in the middle
        def strategy_empty_board():
            if is_board_empty:
                return middle_column

        # best response strategy is also to play ontop of the middle
        def strategy_middle_column():
            if len(played_cols) == 1 and played_cols[0] == middle_column:
                return middle_column
            return middle_column  # always play the middle column if possible

        # if we have a winning move, then play it quick!
        def strategy_connect_four():
            for mark in [hero_mark, villain_mark]:  # win game before blocking villain
                lines_by_length = get_lines_by_length(mark)
                for length, line in lines_by_length:
                    if length == configuration.inarow - 1:  # is one move away from winning
                        values = [ board[cell[0],cell[1]] for cell in line ]
                        middle = get_line_middle(line, mark)
                        if len(middle):
                            for cell in middle:
                                if is_valid_move(cell):
                                    if verbose: print('strategy_connect_four() =', cell, '|', f'mark={hero_mark}v{mark}', values, line, 'middle', get_line_middle(line, mark))
                                    return cell
                        else:
                            edges = get_line_edges(line, mark)
                            for cell in edges:
                                if is_valid_move(cell):
                                    if verbose: print('strategy_connect_four() =', cell, '|', f'mark={hero_mark}v{mark}', values, line, 'middle', get_line_middle(line, mark))
                                    return cell
                    else:
                        break  # get_lines_by_length(mark) is sorted(), so we can short circuit

        # Can we setup a double attack (2x edge_size >= 2)?
        def strategy_double_attack(distance=2):
            for mark in [hero_mark, villain_mark]:
                for length, line in get_lines_by_length(mark):
                    if length >= configuration.inarow - distance:  # line of length 4-2=2
                        values = [ board[cell[0],cell[1]] for cell in line ]
                        playable_edges = set(get_playable_line_edges(line, mark))
                        if not len(playable_edges): continue  # we need a only_playable move

                        zero_groups = get_line_zero_groups(line)
                        if sum(map(len,zero_groups)) < distance: continue  # we need this many cells to complete the line
                        for zero_group in zero_groups:
                            if not len(zero_groups) >= 2: continue          # we need two empty spaces for double attack to work in this direction
                            valid_moves = set(zero_group).intersection(playable_edges)
                            for valid_move in valid_moves:
                                if verbose: print(f'strategy_double_attack({distance}) = {valid_move} | ', f'mark={hero_mark}v{mark}', values, line)
                                return valid_move

        # Can we prepare for a double attack (2x edge_size >= 2)?
        def strategy_prepare_double_attack():
            return strategy_double_attack(distance=3)

        def strategy_intersection_score():
            for mark in [hero_mark, villain_mark]:
                intersections = get_cells_by_intersection_score(mark, only_playable=True)
                for score, cell in intersections:
                    if not is_valid_move(cell): continue
                    if verbose: print(f'strategy_intersection_score() = {cell} | ', f'mark={hero_mark}v{mark}', intersections)
                    return cell

        def strategy_longest_line():
            mark = hero_mark
            for length, line in get_lines_by_length(mark):
                edges = get_playable_line_edges(line, mark)
                for cell in edges:
                    if not is_valid_move(cell): continue
                    if verbose: print(f'strategy_intersection_score() = {cell} | ', f'mark={hero_mark}v{mark}', f'length={length}', line)
                    return cell

        def strategy_random_choice():
            return random.choice(list(valid_cols))


        strategies = [
            strategy_connect_four,
            strategy_empty_board,
            strategy_double_attack,
            # strategy_prepare_double_attack,
            strategy_middle_column,
            strategy_intersection_score,
            strategy_longest_line,
            strategy_random_choice,
        ]
        for strategy in strategies:
            move = strategy()
            if is_valid_move(move):
                print(move, strategy.__name__, 'accepted')
                return move[1] if isinstance(move, tuple) else move
        return None
    except Exception as exception:
        print(type(exception), exception, exception.__traceback__)
        traceback.print_exc()
        raise exception


if __name__ == '__main__':
    my_agent = rules_agent
    from kaggle_environments import make



    env = make("connectx", configuration={
        # "episodeSteps": 0,
        "agentTimeout": 360,
        "actTimeout":   360,
        "runTimeout":   360,
        "rows":    6,
        "columns": 7,
        "inarow":  4,
        "agentExec": "LOCAL",
        "debug": True
    })

    # env.render()
    env.reset()
    # Play as the first agent against default "random" agent.
    # env.run([my_agent, "random"])
    # env.run([my_agent, my_agent])
    env.run([my_agent, 'negamax'])
    # env.render(mode="ipython", width=500, height=450)
    env.render(mode="human")
    #%%





