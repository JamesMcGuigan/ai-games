# %%writefile greenberg_agent.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-greenberg/


# greenberg_agent roshambo bot, winner of 2nd annual roshambo programming competition
# http://webdocs.cs.ualberta.ca/~darse/rsbpc.html

# original source by Andrzej Nagorko
# http://www.mathpuzzle.com/greenberg.c

# Python translation by Travis Erdman
# https://github.com/erdman/roshambo

import random
from operator import itemgetter
# from itertools import izip
izip   = zip   # BUGFIX: izip   is python2
xrange = range # BUGFIX: xrange is python2

rps_to_text  = ('rock','paper','scissors')
rps_to_num   = {'rock':0, 'paper':1, 'scissors':2}

def player(my_moves, opp_moves):
    wins_with    = (1,2,0)  # superior
    best_without = (2,0,1)  # inferior

    lengths = (10, 20, 30, 40, 49, 0)
    p_random = random.choice([0,1,2])  #called 'guess' in iocaine

    TRIALS = 1000
    score_table =((0,-1,1),(1,0,-1),(-1,1,0))
    T = len(opp_moves)  #so T is number of trials completed

    def min_index(values):
        return min(enumerate(values), key=itemgetter(1))[0]

    def max_index(values):
        return max(enumerate(values), key=itemgetter(1))[0]

    def find_best_prediction(l):  # l = len
        bs = -TRIALS
        bp = 0
        if player.p_random_score > bs:
            bs = player.p_random_score
            bp = p_random
        for i in xrange(3):
            for j in xrange(24):
                for k in xrange(4):
                    new_bs = player.p_full_score[T%50][j][k][i] - (player.p_full_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.p_full[j][k] + i) % 3
                for k in xrange(2):
                    new_bs = player.r_full_score[T%50][j][k][i] - (player.r_full_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.r_full[j][k] + i) % 3
            for j in xrange(2):
                for k in xrange(2):
                    new_bs = player.p_freq_score[T%50][j][k][i] - (player.p_freq_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.p_freq[j][k] + i) % 3
                    new_bs = player.r_freq_score[T%50][j][k][i] - (player.r_freq_score[(50+T-l)%50][j][k][i] if l else 0)
                    if new_bs > bs:
                        bs = new_bs
                        bp = (player.r_freq[j][k] + i) % 3
        return bp


    if not my_moves:
        player.opp_history = [0]  #pad to match up with 1-based move indexing in original
        player.my_history = [0]
        player.gear = [[0] for _ in xrange(24)]
        # init()
        player.p_random_score = 0
        player.p_full_score = [[[[0 for i in xrange(3)] for k in xrange(4)] for j in xrange(24)] for l in xrange(50)]
        player.r_full_score = [[[[0 for i in xrange(3)] for k in xrange(2)] for j in xrange(24)] for l in xrange(50)]
        player.p_freq_score = [[[[0 for i in xrange(3)] for k in xrange(2)] for j in xrange(2)] for l in xrange(50)]
        player.r_freq_score = [[[[0 for i in xrange(3)] for k in xrange(2)] for j in xrange(2)] for l in xrange(50)]
        player.s_len = [0] * 6

        player.p_full = [[0,0,0,0] for _ in xrange(24)]
        player.r_full = [[0,0] for _ in xrange(24)]
    else:
        player.my_history.append(rps_to_num[my_moves[-1]])
        player.opp_history.append(rps_to_num[opp_moves[-1]])
        # update_scores()
        player.p_random_score += score_table[p_random][player.opp_history[-1]]
        player.p_full_score[T%50] = [[[player.p_full_score[(T+49)%50][j][k][i] + score_table[(player.p_full[j][k] + i) % 3][player.opp_history[-1]] for i in xrange(3)] for k in xrange(4)] for j in xrange(24)]
        player.r_full_score[T%50] = [[[player.r_full_score[(T+49)%50][j][k][i] + score_table[(player.r_full[j][k] + i) % 3][player.opp_history[-1]] for i in xrange(3)] for k in xrange(2)] for j in xrange(24)]
        player.p_freq_score[T%50] = [[[player.p_freq_score[(T+49)%50][j][k][i] + score_table[(player.p_freq[j][k] + i) % 3][player.opp_history[-1]] for i in xrange(3)] for k in xrange(2)] for j in xrange(2)]
        player.r_freq_score[T%50] = [[[player.r_freq_score[(T+49)%50][j][k][i] + score_table[(player.r_freq[j][k] + i) % 3][player.opp_history[-1]] for i in xrange(3)] for k in xrange(2)] for j in xrange(2)]
        player.s_len = [s + score_table[p][player.opp_history[-1]] for s,p in izip(player.s_len,player.p_len)]


    # update_history_hash()
    if not my_moves:
        player.my_history_hash = [[0],[0],[0],[0]]
        player.opp_history_hash = [[0],[0],[0],[0]]
    else:
        player.my_history_hash[0].append(player.my_history[-1])
        player.opp_history_hash[0].append(player.opp_history[-1])
        for i in xrange(1,4):
            player.my_history_hash[i].append(player.my_history_hash[i-1][-1] * 3 + player.my_history[-1])
            player.opp_history_hash[i].append(player.opp_history_hash[i-1][-1] * 3 + player.opp_history[-1])


    #make_predictions()

    for i in xrange(24):
        player.gear[i].append((3 + player.opp_history[-1] - player.p_full[i][2]) % 3)
        if T > 1:
            player.gear[i][T] += 3 * player.gear[i][T-1]
        player.gear[i][T] %= 9 # clearly there are 9 different gears, but original code only allocated 3 gear_freq's
        # code apparently worked, but got lucky with undefined behavior
        # I fixed by allocating gear_freq with length = 9
    if not my_moves:
        player.freq = [[0,0,0],[0,0,0]]
        value = [[0,0,0],[0,0,0]]
    else:
        player.freq[0][player.my_history[-1]] += 1
        player.freq[1][player.opp_history[-1]] += 1
        value = [[(1000 * (player.freq[i][2] - player.freq[i][1])) / float(T),
                  (1000 * (player.freq[i][0] - player.freq[i][2])) / float(T),
                  (1000 * (player.freq[i][1] - player.freq[i][0])) / float(T)] for i in xrange(2)]
    player.p_freq = [[wins_with[max_index(player.freq[i])], wins_with[max_index(value[i])]] for i in xrange(2)]
    player.r_freq = [[best_without[min_index(player.freq[i])], best_without[min_index(value[i])]] for i in xrange(2)]

    f = [[[[0,0,0] for k in xrange(4)] for j in xrange(2)] for i in xrange(3)]
    t = [[[0,0,0,0] for j in xrange(2)] for i in xrange(3)]

    m_len = [[0 for _ in xrange(T)] for i in xrange(3)]

    for i in xrange(T-1,0,-1):
        m_len[0][i] = 4
        for j in xrange(4):
            if player.my_history_hash[j][i] != player.my_history_hash[j][T]:
                m_len[0][i] = j
                break
        for j in xrange(4):
            if player.opp_history_hash[j][i] != player.opp_history_hash[j][T]:
                m_len[1][i] = j
                break
        for j in xrange(4):
            if player.my_history_hash[j][i] != player.my_history_hash[j][T] or player.opp_history_hash[j][i] != player.opp_history_hash[j][T]:
                m_len[2][i] = j
                break

    for i in xrange(T-1,0,-1):
        for j in xrange(3):
            for k in xrange(m_len[j][i]):
                f[j][0][k][player.my_history[i+1]] += 1
                f[j][1][k][player.opp_history[i+1]] += 1
                t[j][0][k] += 1
                t[j][1][k] += 1

                if t[j][0][k] == 1:
                    player.p_full[j*8 + 0*4 + k][0] = wins_with[player.my_history[i+1]]
                if t[j][1][k] == 1:
                    player.p_full[j*8 + 1*4 + k][0] = wins_with[player.opp_history[i+1]]
                if t[j][0][k] == 3:
                    player.p_full[j*8 + 0*4 + k][1] = wins_with[max_index(f[j][0][k])]
                    player.r_full[j*8 + 0*4 + k][0] = best_without[min_index(f[j][0][k])]
                if t[j][1][k] == 3:
                    player.p_full[j*8 + 1*4 + k][1] = wins_with[max_index(f[j][1][k])]
                    player.r_full[j*8 + 1*4 + k][0] = best_without[min_index(f[j][1][k])]

    for j in xrange(3):
        for k in xrange(4):
            player.p_full[j*8 + 0*4 + k][2] = wins_with[max_index(f[j][0][k])]
            player.r_full[j*8 + 0*4 + k][1] = best_without[min_index(f[j][0][k])]

            player.p_full[j*8 + 1*4 + k][2] = wins_with[max_index(f[j][1][k])]
            player.r_full[j*8 + 1*4 + k][1] = best_without[min_index(f[j][1][k])]

    for j in xrange(24):
        gear_freq = [0] * 9 # was [0,0,0] because original code incorrectly only allocated array length 3

        for i in xrange(T-1,0,-1):
            if player.gear[j][i] == player.gear[j][T]:
                gear_freq[player.gear[j][i+1]] += 1

        #original source allocated to 9 positions of gear_freq array, but only allocated first three
        #also, only looked at first 3 to find the max_index
        #unclear whether to seek max index over all 9 gear_freq's or just first 3 (as original code)
        player.p_full[j][3] = (player.p_full[j][1] + max_index(gear_freq)) % 3

    # end make_predictions()

    player.p_len = [find_best_prediction(l) for l in lengths]

    return rps_to_text[player.p_len[max_index(player.s_len)]]



# observation   =  {'step': 1, 'lastOpponentAction': 1}
# configuration =  {'episodeSteps': 10, 'agentTimeout': 60, 'actTimeout': 1, 'runTimeout': 1200, 'isProduction': False, 'signs': 3}
my_moves    = []
opp_moves   = []
def greenberg_agent(observation, configuration):
    global my_moves
    global opp_moves
    if observation.step > 0:
        opp_move = rps_to_text[ observation.lastOpponentAction ]
        opp_moves.append( opp_move )

    action_text = player(my_moves, opp_moves)
    action      = rps_to_num[action_text]

    my_moves.append(action_text)
    return int(action)
