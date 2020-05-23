import numpy as np

def Recolor(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    N = len(Input)
    list_lenc = []

    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        lenc = len(np.unique(np.array(y)))
        list_lenc.append(lenc)
    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1, 8):
            for Q2 in range(1, 8):
                if Q1 + Q2 == t:
                    Pairs.append((Q1, Q2))
    Pairs.append((15, 15))
    Pairs.append((30, 30))

    for Q1, Q2 in Pairs:
        for v in range(4):

            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}

            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v == 2:
                            p1 = i % Q1
                        else:
                            p1 = (n - 1 - i) % Q1
                        if v == 0 or v == 3:
                            p2 = j % Q2
                        else:
                            p2 = (k - 1 - j) % Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)

                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            # print(Dict)
            if possible:

                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v == 2:
                                p1 = i % Q1
                            else:
                                p1 = (n - 1 - i) % Q1
                            if v == 0 or v == 3:
                                p2 = j % Q2
                            else:
                                p2 = (k - 1 - j) % Q2

                            color1 = x[i][j]
                            rule = (p1, p2, color1)

                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False

                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v

    if Best_Dict == -1:
        return -1  # meaning that we didn't find a rule that works for the traning cases

    # Otherwise there is a rule: so let's use it:
    n = len(Test_Picture)
    k = len(Test_Picture[0])

    answer = np.zeros((n, k), dtype=int)

    for i in range(n):
        for j in range(k):
            if Best_v == 0 or Best_v == 2:
                p1 = i % Best_Q1
            else:
                p1 = (n - 1 - i) % Best_Q1
            if Best_v == 0 or Best_v == 3:
                p2 = j % Best_Q2
            else:
                p2 = (k - 1 - j) % Best_Q2

            color1 = Test_Picture[i][j]
            rule = (p1, p2, color1)
            if (p1, p2, color1) in Best_Dict:
                answer[i][j] = 0 + Best_Dict[rule]
            else:
                answer[i][j] = 0 + color1

            # print(answer)
    # print(list_lenc)
    if len(np.unique(list_lenc)) == 1:
        # print(answer)
        if len(np.unique(answer)) == list_lenc[0]:
            return answer.tolist()
        else:
            return -1

    return answer.tolist()
