# sample_sub3 = pd.read_csv(data_path/ 'sample_submission.csv')
# sample_sub3.head()
#
#
# Solved = []
# Problems = sample_sub3['output_id'].values
# Proposed_Answers = []
# for i in  range(len(Problems)):
#     output_id = Problems[i]
#     task_id = output_id.split('_')[0]
#     pair_id = int(output_id.split('_')[1])
#     f = str(test_path / str(task_id + '.json'))
#
#     with open(f, 'r') as read_file:
#         task = json.load(read_file)
#
#     n = len(task['train'])
#     Input = [Defensive_Copy(task['train'][j]['input']) for j in range(n)]
#     Output = [Defensive_Copy(task['train'][j]['output']) for j in range(n)]
#     Input.append(Defensive_Copy(task['test'][pair_id]['input']))
#
#     #solution = DTSolver([Input, Output])
#   #  solution = solvePatch(np.array(Input[0]),np.array(Output[0]),np.array(Input[-1]))
# #     if solution!=-1:
# #         print(solution)
#        # solution = solvePatch(np.array(Input[0]),np.array(Output[0]),np.array(Input[-1]))
#     predictions=solve_task([Input, Output])
#     pred = ''
#     #predictions=[solution,solution,solution]
#     for i in range(3):
#         if predictions[i] != -1:
#             pred1 = flattener(predictions[i])
#             pred = pred+pred1+' '
#
#         else:
#             pred1 = flattener(example_grid)
#             pred = pred+pred1+' '
#
#     Proposed_Answers.append(pred)
# sample_sub3['output'] = Proposed_Answers
# sample_sub3.to_csv('submission3.csv', index = False)