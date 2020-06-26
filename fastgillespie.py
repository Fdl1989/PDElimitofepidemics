#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:19:36 2020

@author: Francesco Di Lauro
@mail: F.Di-Lauro@sussex.ac.uk
Copyright 2020 Francesco Di Lauro. All Rights Reserved.
See LICENSE file for details

"""
import networkx as nx
import numpy as np
from heapq import *
from datetime import datetime

#0:00:23.466188
class Node():
    def __init__(self,index,status, time):
        self.index = index
        self.status = status
        self.rec_time = time
class Event():
    def __init__(self,node,time,action, source=None):
        self.time = time

        self.node = node
        self.action = action
        self.source=source
    def __lt__(self, other):
        '''
            This is read by heappush to understand what the heap should be about
        '''
        return self.time < other.time        

class fast_Gillespie():
    '''
    This algorithm is inspired by  Joel Miller's algorithm for Fast Gillespie described in the book
    Mathematics of Epidemics on Networks by Kiss, Miller, Simon, 2017, Springer. Section A.1.2 page 384
    '''
    def __init__(self, A, tau=1.0, gamma=2.0, i0=10, tauf=4, discretestep=500):
        if type(A)==nx.classes.graph.Graph:
            self.N = nx.number_of_nodes(A)
            self.A = A
        else:
            raise BaseException("Input networkx object only.")

        # Model Parameters (See Istvan paper).
        self.tau = tau
        self.gamma = gamma
        self.tauf = tauf
        
        # Time-keeping.
        self.cur_time = 0
        #output time vector
        self.time_grid =np.linspace(0,tauf,discretestep)
        self.current_index=0
        
        #Node numbers.
        self.I = np.zeros(discretestep)
        #number of SI links
        self.SI=np.zeros(self.N+1)
        #time in each state
        self.tk = np.zeros(self.N+1)
        
        #node state is [0] if not infected and [1] if infected
        X = np.array([0]*(self.N-i0) +[1]*i0)
        #nodes initialisation
        self.nodes = [Node(i,'susceptible', 0) for i in range(self.N)] 
        #keeps count of how many infected, useful for self.I and self.SI updates
        self.num_I = 0
        #display randomly the initial infected nodes
        np.random.shuffle(X)
        #Queue of Events, here each node has its own event
        self.queue=[]
        self.times=[]
        self.infected=[]
        self.cur_time=0
        for index in np.where(X==1)[0]:
            event = Event(self.nodes[index],0,'transmit', source=Node(-1,'infected',0))
            heappush(self.queue,event)
        
    def run_sim(self):
        '''first round outside to determine SI'''
        num_SI=0        
        while self.queue:
            '''
                condition to stop
            '''
            event = heappop(self.queue)
            #dt is used only to update SI
            '''
            If node is susceptible and it has an event it must be an infection
            '''
            if event.action=='transmit':
                if event.node.status =='susceptible':
                    dt = event.time -self.cur_time
                    #set new time accordingly
        
                    '''
                    check if time grid needs to be updated
                    '''
                    if self.cur_time <self.tauf:
                        while self.time_grid[self.current_index] <= self.cur_time:                    
                            self.I[self.current_index] = self.num_I
                            self.current_index +=1      
                            
                    
                    '''
                    AFTER finding dt you can update SI
                    '''
                    self.SI[self.num_I] += num_SI*dt
                    self.tk[self.num_I] += dt
                    num_SI +=self.process_trans(event.node, event.time)                

                self.find_next_trans(event.source, event.node, event.time)
            else:
                if self.cur_time <self.tauf:
                    while self.time_grid[self.current_index] <= self.cur_time:                    
                        self.I[self.current_index] = self.num_I
                        self.current_index +=1      
                    dt = event.time -self.cur_time
                    self.SI[self.num_I] += num_SI*dt
                    self.tk[self.num_I] += dt
                num_SI +=self.process_rec(event.node,event.time)
        self.I[self.current_index:] = self.I[self.current_index-1]
        
    def process_trans(self,node,time):
        '''
        utility for transmission events:
        it checks also the neighbours.
        Returns number of SI as well
        '''
        #self.times.append(time)
        self.cur_time=time
        self.num_I +=1
        '''
        if len(self.infected) >0:
            self.infected.append(self.infected[-1]+1)
        else:
            self.infected.append(1)
        '''    
        node.status='infected'
        
        r1 = np.random.rand()
        rec_time = time -1.0/self.gamma *np.log(r1)
        node.rec_time = rec_time
        
        if rec_time < self.tauf:
            event = Event(node,rec_time,'recover', None)
            heappush(self.queue,event)
        num_SI=0    
        for index in self.A.neighbors(node.index):
            neighbor = self.nodes[index]
            if neighbor.status=='susceptible':
                num_SI+=1
            else:
                num_SI-=1
            self.find_next_trans(source = node, target = neighbor, time = time)
        return num_SI
    def find_next_trans(self,source,target,time):
        if target.rec_time < source.rec_time:
            r1 = np.random.rand()
            trans_time = max(time,target.rec_time) -1.0/self.tau *np.log(r1)
            if trans_time < source.rec_time and trans_time<self.tauf:
                event = Event(node=target, time=trans_time,  action='transmit', source=source)
                heappush(self.queue,event)
                
    def process_rec(self, node, time):
        node.status='susceptible'
        node.rec_time = 0
        num_SI=0
        self.num_I -=1
        for index in self.A.neighbors(node.index):
            neighbor = self.nodes[index]
            if neighbor.status=='susceptible':
                num_SI-=1
            else:
                num_SI+=1
        #self.times.append(time)
        self.cur_time=time
        #self.infected.append(self.infected[-1]-1)
        return num_SI
                        

if __name__=="__main__":
    N=100000
    k=8
    #A = nx.erdos_renyi_graph(int(N),k/float(N-1.0),seed = 100)
 
    SI_threads=np.zeros(N+1) 
    tk_threads=np.zeros(N+1)   
    startTime = datetime.now()
    #model = fast_Gillespie(A, tau =1, gamma =5, i0 =10)
    #model.run_sim()



    ERgamma = [5, 4.5, 7]
    ERtau =[1, 1, 4]
    ERk = [8.0,10.0,7.0]
    taufv = [4,3,0.8]

    number_of_networkgen = 100
        
    number_of_epid = 200
    from datetime import datetime
    from matplotlib import pyplot as plt
    startTime = datetime.now()

    for i in range(3):
        for N in [1000,100000]:
            k = ERk[i]
            gamma = ERgamma[i]
            tau = ERtau[i]   
            tauf = taufv[i]
            R0_er = tau*(k*(N-2)/(N-1.0))/(tau+gamma)
            print(R0_er)    
            #print(tauf)
            networkchoice='E-R'   
            seed_vector=np.array([j*(124)+23 for j in range(number_of_networkgen*number_of_epid)])
            
            A =  nx.fast_gnp_random_graph(N,k/float(N-1.0))
            fig = plt.figure()
            for j in range(5):
                
                model = fast_Gillespie(A, tau =tau, gamma =gamma, i0 =10)
                model.run_sim() # Run the simulation.
                plt.plot(model.time_grid,model.I)
                model = fast_Gillespie(A,tau=tau, gamma=gamma, i0=N)
                model.run_sim()                                    
                plt.plot(model.time_grid,model.I)
            plt.title("k=%d"%k)
            plt.show()
    '''    
    for i in range(10):
        model = fast_Gillespie(A, tau =1, gamma =5, i0 =10)
        model.run_sim()
        SI_threads += model.SI
        tk_threads += model.tk
    for i in range(10):
        model = fast_Gillespie(A, tau =1, gamma =5, i0 =100)
        model.run_sim()
        SI_threads += model.SI
        tk_threads += model.tk 
       #plt.plot(model.time_grid, model.I)
    print(datetime.now()-startTime)
    import matplotlib.pyplot as plt
    tk_threads[np.where(tk_threads==0)]=1
    plt.plot(np.arange(0,101), SI_threads/tk_threads)
    #plt.plot(model.time_grid, model.I)                
    #plt.show()
    
    #I = np.arange(1001)

    #avg_SI= model.SI/model.tk
    #plt.plot(I,avg_SI)
    '''

                
