# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:30:13 2022

@author: NitheshV, SarangK, SourabhK
"""
#%% Block1: All the libraries required are imported here
import numpy as np
import time

#%% Block2: The Adjacency matrix for a given TSP is defined as matrix below
adj_mat = np.array([[0,5,8,0,0,0,0,0],
            [0,0,4,0,4,0,0,0],
            [0,0,0,2,0,0,5,0],
            [0,0,0,0,0,0,0,7],
            [5,0,0,0,0,0,0,0],
            [0,6,0,0,2,0,0,0],
            [0,0,0,3,0,8,0,0],
            [0,0,0,0,0,5,4,0]])

#%% Block3: Adjacency list is extracted from a given adjacency matrix
def adj_list_fcn(adj_mat):
    adj_list = {}
    for row in range(adj_mat.shape[0]):
        adj_list[row] = list(np.nonzero(adj_mat[row])[0])
    return adj_list

#%% Block4: A minimum cost is computed for the current choice of traversal order on the reduced adjacency matrix by excluding all non-feasible states
def cost_fcn(adj_mat):
    r_min = 0
    for row in range(adj_mat.shape[0]):
        r_tmp = adj_mat[row]
        r_tmp = r_tmp[np.nonzero(r_tmp)]
        try:
            r_min = np.min(r_tmp) + r_min
        except:
            continue                 
    return r_min

#%% Block5: This functions finds all  the combination of pairs of nodes from the given traversal order whose weights/cost are not considered during the calculation of minimum cost
def skip_nodes_fcn(adj_mat,stack):
    org_set = np.arange(adj_mat.shape[0])
    novisit_nodes = list(set(org_set) ^ set(stack))
    inv_stack = list(np.fliplr([stack])[0])
    skip_nodes_list = [[stack[-1],stack[0]]]
    for i in novisit_nodes:
        for j in inv_stack:
            if j is not inv_stack[-1]:
              skip_nodes_list.append([i,j])
    return skip_nodes_list

#%% Block6: The adjacency matrix is reduced by zeroing all the nodes that are not to be considered for a given traversal of choice
def updt_tmp_matrix(skip_nodes_list, adj_mat):
    for i in skip_nodes_list:
        adj_mat[i[0]][i[1]] = 0
    return adj_mat

#%% Block7: This function extracts the key with the help of a value from a dictionary
def dict_decode(dictionary, value):
    keys = list(dictionary.keys())
    vals = list(dictionary.values())
    pos = vals.index(value)
    key = keys[pos]
    return key
#%%
#The code that follows below implements the main algorithm to solve a TSP.    

#%% Block8: Initialization of all variables, lists, arrays and dictionaries that are used in the algorithm.
adj_list = adj_list_fcn(adj_mat)
start_node = 0
init_node = start_node
cost_stack = {}
visited_nodes = [start_node]
history = {}
finFlag = False
new_stack = [start_node]
traversal_history = {}
pos = 0
start_time = time.time()

#%% Block9:  The bulk of the TSP algorithm is implemented here that iterates over all possible combinations of the traversal order for a given adjacency matrix until an optimal solution is found and breaks out from the loop to terminate the search. There are several sub blocks inside this block.

while ~finFlag:
   #%%  Block 9.1:  Computing the cost for all linked nodes of the last node in the current traversal order
    for linkd_node in adj_list[start_node]:
        curr_cost = adj_mat[start_node][linkd_node] 
        tmp_adj_mat = np.copy(adj_mat)
        dummy_stack = list(np.copy(new_stack))
        if linkd_node in dummy_stack:
            continue
        dummy_stack.append(linkd_node)
        skip_nodes_list = skip_nodes_fcn(adj_mat,dummy_stack)
        tmp_adj_mat = updt_tmp_matrix(skip_nodes_list, tmp_adj_mat)
        cost_stack[linkd_node] = cost_fcn(tmp_adj_mat)
    print('Cost_stack {}'.format(cost_stack))
    
    #%%  Block 9.2: Switching to next promising node in history as no promising node is available in the current search
    if not bool(cost_stack):
        new_stack = new_stack[0:pos]
        start_node = dict_decode(history, np.min([history[x] for x in history]))
        new_stack.append(start_node)
        del history[start_node]
        print('----------------------')
        print('Backtracking to more promising node and continuing the search')
        print(new_stack)
        continue
    
    #%%  Block 9.3: Traversal stack is updated and history of the unexplored states is recorded in case there are no more promising nodes from current traversal order
    old_start_node = start_node
    start_node = dict_decode(cost_stack,np.min([cost_stack[x] for x in cost_stack]))
    unexp_nodes = [j for j in adj_list[old_start_node] if j not in new_stack]
    visited_nodes.append(start_node)
    for k in unexp_nodes:
        if k not in visited_nodes:
            history[k] = cost_stack[k]
            pos = pos + 1
    new_stack.append(start_node)
    if bool(history):
        min_history_cost = np.min([history[x] for x in history])

    #%%  Block 9.4: Traversal order search is completed and unexplored history is empty or current cost is less than all other costs in the history
    if len(new_stack) == adj_mat.shape[0] and (not bool(history) or np.min([cost_stack[x] for x in cost_stack]) <= min_history_cost):
        finFlag = True
        print('-------------------')
        print(new_stack)
        new_stack.append(new_stack[0])
        print('Search is terminated')
        print('*')
        print('*')
        print('Outcome is as follows -->')
        print('Optimal Traversal Order for the given TSP: {}'.format(new_stack))
        print('Cost of the above traversal order: {}'.format(np.min(cost_stack[start_node]) + adj_mat[start_node][init_node]))
        start_node = init_node
        end_time = time.time()
        print('Total time taken to complete the search is {:0.10f}'.format(end_time - start_time))
        break
    
  #%%  Block 9.5: More promising node is available in the unexplored history than the last node in current traversal order search
    elif len(new_stack) == adj_mat.shape[0] and np.min([cost_stack[x] for x in cost_stack]) > np.min([history[x] for x in history]):
        finFlag = False
        start_node = dict_decode(history, min_history_cost)
        traversal_history[np.min([cost_stack[x] for x in cost_stack])] = new_stack
        new_stack = new_stack[0:(new_stack.index(start_node)-1)]
        new_stack.append(start_node)
        del history[start_node]
        print('Backtracking to more promising node and continuing the search')
        
   #%% Block 9.5:  Continue search with the current traversal order as it is more optimal
    else:
        finFlag = False
        start_node = dict_decode(cost_stack, np.min([cost_stack[x] for x in cost_stack]))
    
    
   #%% Block 9.6:  ReInitialization
    cost_stack = {}
    print('-------------------')
    print(new_stack)
