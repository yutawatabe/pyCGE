#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This module builds EK model as a class. Basic functions 
"""

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns

class onesector():

    def __init__(self,
                
        # Metadata
        year, # Year of the data
        countries, # List of ISO3 codes

        # Equilibrium outcome
        X, # Trade flows (N * N ndarray)

        # Fundamental parameters
        theta = 4, # Trade elasticity (in this model, this is sigma -1)  
        ):
 
        # Set country list and year and model
        self.year,self.countries = year,countries
        self.N = len(countries)

        # Set parameters
        self.theta,self.X = theta, X

        # Calculate absorption, production, trade deficit
        self.Xm = np.sum(X,axis=(0))
        self.Ym = np.sum(X,axis=(1))
        self.D  = self.Xm - self.Ym 

        return

    @staticmethod
    def from_static_parameters(
        # Metadata
        countries, # List of ISO3 codes: list(N)
        year, # Year of the data we use: int 
        
        # Elasticity
        theta,  # Trade elasticity: ndarray(S)
        
        # Deep parameters
        A, # Technology parameter: ndarray(N)
        tau, # Trade cost parameter: ndarray(N,N)
        L, # Labor endowment: ndarray(N)
        D, # Trade deficit: ndarray(N))
    ):    
        
        """
        This method calculates a static equilibrium from the parameters.
        """

        # Set some convergence parameter
        psi = 0.1 # Convergence speed
        tol = 0.000001 # Convergence tolerance
        dif = 1 # initial convergence criterion 

        # Start solving the eqm
        N = len(countries)
        w = np.ones(N)

        # Normalize this...
        wgdp = np.sum(w * L)
        w = w / wgdp

        while dif > tol: 
            # Update w and r
            w_old = np.copy(w)

            # Calculate price
            p = np.zeros((N,N))
            for OR,DE in np.ndindex((N,N)):
                p[OR,DE] = w[OR]* tau[OR,DE] / A[OR] 
            
            # Calculate pi
            pi_num = np.zeros((N,N))
            pi_den = np.zeros((N))
            pi = np.zeros((N,N))
            for OR,DE in np.ndindex((N,N)):
                pi_num[OR,DE] = p[OR,DE]**(-theta)
                pi_den[DE] += pi_num[OR,DE]
            for OR,DE in np.ndindex((N,N)):
                pi[OR,DE] = pi_num[OR,DE] / pi_den[DE]

            # Calculate price index
            P = pi_den**(-1/theta)
            
            # Calculate excess factor demand
            wLS = w * L
            Xm  = wLS + D

            wLD = np.zeros((N))
            for OR,DE in np.ndindex((N,N)):
                wLD[OR] += pi[OR,DE] * Xm[DE]
            ZL = (wLD - wLS) / w
            w = w * (1 + psi / L * ZL)

            dif = max(np.max(np.abs(w - w_old)))

            wgdp = np.sum(w * L )
            w = w / wgdp

        Xm = w * L + D
        X = np.zeros((N,N))
        for OR,DE in np.ndindex((N,N)):
             X[OR,DE] = pi[OR,DE] * Xm[DE]
        wL = w * L
        return onesector(year=year,countries=countries,X=X,theta=theta)
    
