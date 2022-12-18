# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:36:30 2022

@author: Nithesh
"""

import numpy as np
#%%
class hmmRobLoc:
    #Constructor Initialization
    def __init__(self):
        self.dist = (100/38)*np.ones((6,7)) # Initializing the prior distribution for the maze
        self.dist[1][1] = 0 #Obstacle1
        self.dist[1][4] = 0 #Obstacle2
        self.dist[3][1] = 0 #Obstacle3
        self.dist[3][4] = 0 #Obstacle4
        #All observations in even and motions in odd positions
        self.obvactSpace = [[0,0,0,0],'N',[1,0,0,0],'N',[0,0,0,0],'W',[0,1,0,1],'W',[1,0,0,0]] 
        self.dim = self.dist.shape[0]*self.dist.shape[1] # Dimension for transition matrix computation
         
    
    #To fetch the real state in all directions
    def actualObv(self, dist, row,col):
        realObv = []
        
        # Looking towards West by decreasing the column count by 1
        try:
            if col - 1 < 0 or dist[row][col - 1] == 0:
                realObv.append(1)
            elif col - 1 >= 0:
                realObv.append(0) 
        except:
            realObv.append(1)
            
        # Looking towards North by decreasing the row count by 1
        try:
            if row - 1 < 0 or dist[row - 1][col] == 0:
               realObv.append(1)  
            elif row - 1 >= 0:
                realObv.append(0)
        except:
            realObv.append(1)
         
        # Looking towards East  by increasing the column count by 1
        try:
            if col + 1 > dist.shape[1] or dist[row][col + 1] == 0:
                realObv.append(1)
            elif col + 1 <= dist.shape[1]:
                realObv.append(0)
        except:
            realObv.append(1)
        
        # Looking towards South by increasing the row count by 1
        try:
            if row + 1 > dist.shape[0] or dist[row + 1][col] == 0:
               realObv.append(1)  
            elif row + 1 <= dist.shape[0]:
                realObv.append(0)
        except:
            realObv.append(1)
        return realObv
    
    # Calculates the likelihood by comparing the real state and sensor observations
    # in all directions and taking product of all the probabilities
    def likelihoodCal(self, realObv, senObv):
        initprob = 1
        for k in range(len(realObv)):
            # Given Obstacle, sensed an obstacle
            if realObv[k] == 1 and senObv[k] == 1:
                prob = 0.8
                
            # Given an empty space, sensed an obstacle
            elif realObv[k] == 0 and senObv[k] == 1:
                prob = 0.15
                
            # Given an obstacle, sensed an empty space
            elif realObv[k] == 1 and senObv[k] == 0:
                prob = 0.2
                
            # Given an empty space, sensed an empty space    
            elif realObv[k] == 0 and senObv[k] == 0:
                prob = 0.85

            prob = initprob*prob
            initprob = prob      
        return prob
    
    # Calculating the transformation matrix for motion in W,N,E and S
    def tMatCal(self, dist, act):
        if act == 'N':
            actspace = [0.1,0.8,0.1,0]
        elif act == 'W':
            actspace = [0.8,0.1,0,0.1]
        elif act == 'E':
            actspace = [0,0.1,0.8,0.1]
        elif act == 'S':
            actspace = [0.1,0,0.1,0.8]
        
        #Initializing the Transition matrix to zero
        tmat = np.zeros((self.dim,self.dim))
        xx = 0 
        yy = 0
        #given state
        for row in range(self.dim):
            x = np.mod(row, dist.shape[1]) #axis 1 of given state
            yy = 0
            if row > 0 and x == 0 and xx < 5: # To increment the row
                xx +=1 #axis 0 of given state
            
            # When an obstacle is the given state, simply ignore
            if dist[xx][x] == 0:
                  continue
            
            #sweeping through all other states to check the probability of reaching
            #from a given state
            for col in range(self.dim):
                y = np.mod(col, dist.shape[1]) #axis 1 of destination state
                if  col > 0 and y == 0 and yy < 6: # To increment the row
                    yy +=1 #axis 0 of destination state

                # case1: Westward motion from given state
                if xx == yy and x - y == 1:
                    tmat[row][col] = actspace[0]
                 
                # case2: Eastward motion from given state
                if xx == yy and x - y == -1:
                    tmat[row][col] = actspace[2]
                
                # case3: Northward motion from given state
                if x == y and xx - yy == 1:
                    tmat[row][col] = actspace[1]
                
                # case4: Southward motion from given state
                if x == y and xx - yy == -1:
                    tmat[row][col] = actspace[3]
                    
                # case5: Stay put by bouncing back hitting a wall
                if (xx == yy and x == y):
                    realObv = self.actualObv(dist,xx,x)
                    initval = 0
                    for i in range(len(realObv)):
                        val = initval + realObv[i]*actspace[i]
                        initval = val
                    tmat[row][col] = val
                    
                # case6: Bump into an obstacle
                if dist[yy][y] == 0:
                    tmat[row][col] = 0
                    
        return tmat           
                    
    # Main HMM algorithm
    def hmmAlgo(self):
        for i in range(len(self.obvactSpace)):
            
            #Filtering(every even element in the observation action space defined in the problem)
            if np.mod(i,2) == 0:
                # Sweeping through every state to calculate the posterior density
                for row in range(self.dist.shape[0]):
                    for col in range(self.dist.shape[1]):
                        realObv = self.actualObv(self.dist,row,col)
                        senObv = self.obvactSpace[i]
                        prob = self.likelihoodCal(realObv, senObv)
                        self.dist[row][col] = self.dist[row][col]*prob #Likelihood*prior = psoterior
                step = 'Sensing'
                
                #normalization
                self.dist = self.dist/np.sum(np.sum(self.dist))*100
                print('-------------------------------------------------------------')  
                print('Current step: {}'.format(step))
                print(self.dist)
                print(np.sum(np.sum(self.dist)))
        
            #Prediction(every odd element in the observation action space defined in the problem)
            elif np.mod(i,2) == 1: 
                # Transition Matrix
                tmat = self.tMatCal(self.dist,self.obvactSpace[i]) 
                # Flattening the posterior matrix from filtering step
                self.dist = self.dist.flatten() #flattening the posterior matrix from filtering step
                # Dot product of flattened matrix with Transition matrix
                self.dist = np.dot(self.dist,tmat) 
                # Reshaping the dot product to size of the maze
                self.dist = self.dist.reshape((6,7)) 
                step = 'Motion'
                print('-------------------------------------------------------------')  
                print('Current step: {}'.format(step))
                print(self.dist)
                print(np.sum(np.sum(self.dist)))
#%% Calling the Main Algorith with an instantiation of the class.
robloc = hmmRobLoc()
robloc.hmmAlgo()
