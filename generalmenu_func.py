#!/usr/bin/env python3
"""{{{1
Solve for menu costs equilibria.
}}}1"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import copy
import datetime
import functools
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.optimize import fminbound 
import shutil

# Defaults:{{{1
# SIGMA = 4 # matches Nakamura Steinsson 2008
SIGMA = 8 # matches Nakamura Steinsson 2008
keydetails = ['lowpointprob', 'highpointprob', 'price_change_size_pos', 'price_change_size_neg', 'price_change_size_abs', 'price_change_prob_pos', 'price_change_prob_neg', 'price_change_prob', 'cesrelprice', 'cesrelprice_nopower', 'NU', 'MC', 'aggMC', 'profitshare', 'profitsharemenu', 'menu', 'menushare', 'pistar', 'BETA', 'SIGMA', 'polfirst', 'polmedian', 'pollast']
# make starttime a global variable
starttime = datetime.datetime.now()
betas_list_default = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.97, 0.98, 0.99]
sigmas_list_default = [4, 5, 6, 7, 8, 9, 10]

# Solve Value Function:{{{1
def vfidiscrete(profitarray, endogstate, transmissionarray, beta, menucost, inflation, crit = 1e-5, printinfo = False, basicchecks = True, logendog = False, returnpol = True):
    """
    Doesn't really make sense to use this with zero inflation. Better to use vfidiscrete_noinflation then.

    profitarray.shape = ns1 x ns2
    transmissionarray = ns2 x ns2 where (i,j) is P(ns2' = j|ns2 = i)
    (V = ns1 x ns2). So (i,j) of V is V(s1 = ith element, s2 = jth element)
    """

    ns1 = profitarray.shape[0]
    ns2 = profitarray.shape[1]

    if basicchecks is True:
        # check the transmission probabilities make sense
        for state in range(ns2):
            if abs(np.sum(transmissionarray[state][:]) - 1) > 1e6:
                raise ValueError('ERROR: transmissionarray rows not sum to 1. State: ' + str(state) + '. Sum: ' + str(np.sum(transmissionarray[state][action][:])) + '.')
            

    V = np.zeros([ns1, ns2])
    Vp_change_vector = np.zeros([ns2])
    Vp = copy.deepcopy(V)
    Vp_change = copy.deepcopy(V)
    Vp_nochange = copy.deepcopy(V)

    pol_change_vector = np.zeros([ns2])
    pol_change_vector = pol_change_vector.astype(int)

    if returnpol is True:
        # pol must be integers
        pol = np.empty([ns1, ns2])
        price_change = np.empty([ns1, ns2])

    # preliminary steps I need to take when inflation != 0
    if inflation != 0:
        # for a given old relative price, get which value to use for the value of not changing relative price today for the future
        if logendog is True:
            raise ValueError('Need to add this.')
        else:
            endogstate_unchanged = endogstate / (1 + inflation)
            sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
            from dist_func import weightvalue_orderedlist
            firstnonzeroweightindex, weightonfirst, firstindexonly, lastindexonly = weightvalue_orderedlist(endogstate_unchanged, endogstate)
            firstnonzeroweightindex_p1 = [index + 1 for index in firstnonzeroweightindex]
            # make weightonfirst a list
            weightonfirst = np.array(weightonfirst)

    iterationi = 1
    while True:
        # if change
        for s2 in range(0, ns2):
            maxval = profitarray[:, s2] + beta*V.dot(transmissionarray[s2, :]) - menucost
            pol_change_vector[s2] = np.argmax(maxval)
            Vp_change_vector[s2] = maxval[pol_change_vector[s2]]
        Vp_change = np.tile(Vp_change_vector, (ns1, 1))

        if inflation != 0:
            # if no change in absolute price
            # need to get V(p/Pistar, exog)
            Vp_firstnonzeroweight = V[firstnonzeroweightindex, :]
            Vp_firstnonzeroweight_p1 = V[firstnonzeroweightindex_p1, :]
            # get weighted average of Vp_overPistar for points in between endogstate bounds
            # [[1,2], [3,4]] * np.array([0.1, 0.5]).reshape(2, 1) = [[0.1,0.2], [1.5,2]]
            # thus I get a weight of Vp between p/Pistar below and above it
            middleelements = Vp_firstnonzeroweight * np.array(weightonfirst).reshape(len(weightonfirst), 1) + Vp_firstnonzeroweight_p1 * np.array(1 - weightonfirst).reshape(len(weightonfirst), 1)
            if firstindexonly > 0:
                firstelements = np.tile(V[0, :], (firstindexonly, 1))
            else:
                firstelements = np.empty([0, ns2])
            if lastindexonly > 0:
                lastelements = np.tile(V[-1, :], (lastindexonly, 1))
            else:
                lastelements = np.empty([0, ns2])
            V_overPistar = np.concatenate((firstelements, middleelements, lastelements), axis = 0)
        else:
            V_overPistar = V

        for s2 in range(0, ns2):
            Vp_nochange[:, s2] = profitarray[:, s2] + beta*V_overPistar.dot(transmissionarray[s2, :])

        # get Vp
        Vp = np.maximum(Vp_change, Vp_nochange)

        diff = np.max(np.abs(Vp - V))
        if np.isnan(diff):
            print('ERROR: diff is nan on iteration ' + str(iterationi))
            sys.exit(1)
        if printinfo is True:
            print('Iteration ' + str(iterationi) + '. Diff: ' + str(diff) + '.')
        iterationi = iterationi + 1
        if diff < crit:
            break
        else:
            # need copy otherwise when replace Vp[s], V[s] also updates
            V = Vp.copy()

    # get pol if desired
    if returnpol is True:
        for s1 in range(0, ns1):
            for s2 in range(0, ns2):
                if Vp_change[s1, s2] < Vp_nochange[s1, s2]:
                    # set pol to be -1 if no change in price
                    pol[s1, s2] = endogstate[s1] / (1 + inflation)
                    price_change[s1, s2] = 0
                else:
                    pol[s1, s2] = endogstate[pol_change_vector[s2]]
                    price_change[s1, s2] = 1

        return(Vp, pol, price_change)
    else:
        return(Vp)


def vficontinuous(profitfunction, endogstatevec, exogstatevec, transmissionarray, beta, menucost, inflation = 0, crit = 0.000001, printinfo = False, basicchecks = True, returnpol = True):
    """
    Solve for the partial equilibrium continuous solution.
    Solve for the policy function and whether price changes by assuming only discrete states of p_i are available.

    Method of solving for policy function does not work so well if I want to find a continuous solution because there will be discrete mass jumps when prices shift from changing to not changing.

    Precise to within degree of p_i so if p_i 0.001 apart then should be that precise at least
    """
    ns1 = len(endogstatevec)
    ns2 = np.shape(transmissionarray)[0]
    
    if basicchecks is True:
        # check the transmission probabilities make sense
        for state in range(ns2):
            if abs(np.sum(transmissionarray[state][:]) - 1) > 1e6:
                raise ValueError('ERROR: transmissionarray rows not sum to 1. State: ' + str(state) + '. Sum: ' + str(np.sum(transmissionarray[state][action][:])) + '.')
            

    # initial values
    V = np.zeros([ns1, ns2])
    Vp_change_vector = np.zeros([ns2])
    Vp = copy.deepcopy(V)
    Vp_nochange = copy.deepcopy(V)
    pol_change_vector = np.zeros([ns2])
    pol_nochange = np.zeros([ns1, ns2])
    if returnpol is True:
        price_change = np.zeros([ns1, ns2])
        price_change = price_change.astype(int)
        pol = np.zeros([ns1, ns2])


    # define negative value function
    def negativevaluefunction(betaEVfunc, s2_val, s1_new_val):
        value = profitfunction(s1_new_val, s2_val) + betaEVfunc(s1_new_val)
        return(-value)
        
    iterationi = 1
    while True:
        for s2 in range(ns2):

            # compute expected value function
            betaEV = beta*Vp.dot(transmissionarray[s2, :])
            # using interp from numpy - seems to be quicker than scipy
            def betaEVfunc(s1_new_val):
                return(np.interp(s1_new_val, endogstatevec, betaEV))
            # betaEVfunc = interp1d(endogstatevec, betaEV)


            # CHANGE PRICE
            # get current value function
            currentnegvalfunc = functools.partial(negativevaluefunction, betaEVfunc, exogstatevec[s2])
            # get s1new that maximises utility
            pol_change_vector[s2] = fminbound(currentnegvalfunc, endogstatevec[0], endogstatevec[-1])
            Vp_change_vector[s2] = -currentnegvalfunc(pol_change_vector[s2]) - menucost


            # NO CHANGE PRICE
            for s1 in range(ns1):
                # if previous price set was 1 then relative price when unchanged is 1/(1 + inflation)
                pol_nochange[s1, s2] = endogstatevec[s1] / (1 + inflation)
                Vp_nochange[s1, s2] = -negativevaluefunction(betaEVfunc, exogstatevec[s2], pol_nochange[s1, s2])

        Vp_change = np.tile(Vp_change_vector, (ns1, 1))

        Vp = np.maximum(Vp_change, Vp_nochange)

        diff = np.max(np.abs(Vp - V))
        if np.isnan(diff):
            print('ERROR: diff is nan on iteration ' + str(iterationi))
            sys.exit(1)
        if printinfo is True:
            print('Iteration ' + str(iterationi) + '. Diff: ' + str(diff) + '.')
        iterationi = iterationi + 1
        if diff < crit:
            break
        else:
            # need copy otherwise when replace Vp[s], V[s] also updates
            V = Vp.copy()

    if returnpol is True:
        # get pol and price_change by finding whether best to change price or not
        for s1 in range(ns1):
            for s2 in range(ns2):
                if Vp_change_vector[s2] < Vp_nochange[s1, s2]:
                    # set pol to be -1 if no change in price
                    pol[s1, s2] = pol_nochange[s1, s2]
                    price_change[s1, s2] = 0
                else:
                    pol[s1, s2] = pol_change_vector[s2]
                    price_change[s1, s2] = 1

        return(Vp, pol, price_change)
    else:
        return(Vp)


# Functions to Solve for Policy Functions More Precisely:{{{1
def getpol_quick(profitfunction, endogstatevec, exogstatevec, transmissionarray, beta, menucost, inflation, V):
    """
    Want to solve for smooth policy function i.e. don't want discrete jumps in mass when price shifts from changing to not changing.
    To do this, if one discrete price does not change and the next does, set the price chosen by firms on these discrete points to be somewhere between the actual prices chosen.

    Advantage: Easy to apply my usual method for solving for the transmissionstararray since each state is associated with a specific chosen price.
    Disadvantage: This is a bit messy and doesn't appear to solve continuously as I hoped.
    So I wrote NAME which properly solves for the policy function but requires using a different method to get transmissionstararray.
    """
    ns1 = len(endogstatevec)
    ns2 = np.shape(transmissionarray)[0]
    
    # initial values
    pol = np.zeros([ns1, ns2])
    pol_change_vector = np.zeros([ns2])
    price_change = np.zeros([ns1, ns2])
    Vp_change_vector = np.zeros([ns2])
    pol_nochange = copy.deepcopy(pol)
    Vp_nochange = copy.deepcopy(V)

    # define negative value function
    def negativevaluefunction(betaEVfunc, s2_val, s1_new_val):
        value = profitfunction(s1_new_val, s2_val) + betaEVfunc(s1_new_val)
        return(-value)
        
    for s2 in range(ns2):

        # compute expected value function
        betaEV = beta*V.dot(transmissionarray[s2, :])
        # using interp from numpy - seems to be quicker than scipy
        def betaEVfunc(s1_new_val):
            return(np.interp(s1_new_val, endogstatevec, betaEV))
        # betaEVfunc = interp1d(endogstatevec, betaEV)


        # CHANGE PRICE
        # get current value function
        currentnegvalfunc = functools.partial(negativevaluefunction, betaEVfunc, exogstatevec[s2])
        # get s1new that maximises utility
        pol_change_vector[s2] = fminbound(currentnegvalfunc, endogstatevec[0], endogstatevec[-1])
        Vp_change_vector[s2] = -currentnegvalfunc(pol_change_vector[s2]) - menucost


        # NO CHANGE PRICE
        for s1 in range(ns1):
            pol_nochange[s1, s2] = endogstatevec[s1] / (1 + inflation)
            Vp_nochange[s1, s2] = -negativevaluefunction(betaEVfunc, exogstatevec[s2], pol_nochange[s1, s2])


    # example: p_i are 0.98, 0.99, 1.00, 1.01, 1.02
    # basic idea here is that we are capturing value function based upon prob of whether change price in ranges <0.985, 0.985-0.995, 0.995-1.005, 1.005-1.015, >1.015 so that get continuous range
    for s2 in range(ns2):
        # lowsideprobs are probability that <0.98 , 0.985-0.990, 0.995-1.00, 1.05-1.10, 1.15-1.20 change price
        # based upon comparison of each point with the point beneath it (except first point)
        for s1 in range(0, len(endogstatevec)):
            if s1 == 0:
                # to avoid introducing discontinuity, set that lowsideprobchange = 1 for lowest state always
                # we know this is likely to hold anyway since with an extreme price to begin, you're more likely to want to change it
                lowsideprobchange = 0.5
            else:
                # take value that got in previous iteration
                lowsideprobchange = nextlowsideprobchange
            if s1 == len(endogstatevec) - 1:
                # again to ensure continuity, assume that highsideprobchange for highest state is 1
                # again we know this is likely to hold anyway since with an extreme price to begin, you're more likely to want to change it
                highsideprobchange = 0.5
            elif Vp_change_vector[s2] < Vp_nochange[s1, s2] and Vp_change_vector[s2] < Vp_nochange[s1 + 1, s2]:
                highsideprobchange = 0.0
                nextlowsideprobchange = 0.0
            elif Vp_change_vector[s2] >= Vp_nochange[s1, s2] and Vp_change_vector[s2] >= Vp_nochange[s1 + 1, s2]:
                highsideprobchange = 0.5
                nextlowsideprobchange = 0.5
            elif Vp_change_vector[s2] >= Vp_nochange[s1, s2] and Vp_change_vector[s2] < Vp_nochange[s1 + 1, s2]:
                # Vp_change = 1; Vp_nochange for s1 == 0 is 0.95, Vp_nochange for s1 == 1 is 1.1
                # intuitively we see that we want to model it like at 0.333 the agent shifts from picking change to nochange
                # since this is less than 0.5, all nextlowsideprobchange for s1 == 1 is 0
                # split highsideprobchange between nochange and change proportionately
                pointwhenstopchange = (Vp_change_vector[s2] - Vp_nochange[s1, s2]) / (Vp_change_vector[s2] - Vp_nochange[s1, s2] +  Vp_nochange[s1 + 1, s2] - Vp_change_vector[s2])
                if pointwhenstopchange < 0.5:
                    highsideprobchange = pointwhenstopchange
                    nextlowsideprobchange = 0.0
                else:
                    highsideprobchange = 0.5
                    nextlowsideprobchange = pointwhenstopchange - 0.5
            elif Vp_change_vector[s2] < Vp_nochange[s1, s2] and Vp_change_vector[s2] >= Vp_nochange[s1 + 1, s2]:
                # Vp_change = 1; Vp_nochange for s1 == 1 is 1.1, Vp_nochange for s1 == 2 is 0.95
                # intuitively we see that we want to model it like at 1.666 the agent shifts from picking nochange to change
                # since this is greater than 1.5, all highsideprobchange for s1 == 1 is 0
                # split nextlowsideprobchange between nochange and change proportionately
                pointwhenstartchange = (Vp_nochange[s1, s2] - Vp_change_vector[s2]) / (Vp_nochange[s1, s2] - Vp_change_vector[s2] + Vp_change_vector[s2] - Vp_nochange[s1 + 1, s2])
                if pointwhenstartchange < 0.5:
                    highsideprobchange = 0.5 - pointwhenstartchange
                    nextlowsideprobchange = 0.5
                else:
                    highsideprobchange = 0.0
                    nextlowsideprobchange = 1 - pointwhenstartchange
            else:
                raise ValueError('')

            probchange = lowsideprobchange + highsideprobchange

            pol[s1, s2] = probchange * pol_change_vector[s2] + (1 - probchange) * pol_nochange[s1, s2]
            # ensure not smaller than value you could get from changing your price
            # Vp[s1, s2] = np.max([ probchange * Vp_change_vector[s2] + (1 - probchange) * Vp_nochange[s1, s2], Vp_change_vector[s2] ])
            price_change[s1, s2] = probchange

    return(pol, price_change)

def getpol(profitfunction, endogstatevec, exogstatevec, transmissionarray, beta, menu, inflation, V):
    """
    Inputs: The solution for the value function under different prices and producitivities.
    This function: Solves for a more precise policy function returning for each productivity, the optimal reset price and the previous price bands under which agents do not change their price.

    Note that if I have the relative price for the current period, I will need to multiply it by 1 + inflation before examining whether I change price or not since the bounds are on the previous period price.

    """

    # policy function
    pol = []
    ns2 = len(exogstatevec)

    # define negative value function
    def negativevaluefunction(betaEVfunc, s2_val, s1_new_val):
        value = profitfunction(s1_new_val, s2_val) + betaEVfunc(s1_new_val)
        return(-value)
        
    for s2 in range(ns2):

        # compute expected value function
        betaEV = beta*V.dot(transmissionarray[s2, :])
        # using interp from numpy - seems to be quicker than scipy
        def betaEVfunc(s1_new_val):
            return(np.interp(s1_new_val, endogstatevec, betaEV))
        # betaEVfunc = interp1d(endogstatevec, betaEV)


        # CHANGE PRICE
        # get current value function
        currentnegvalfunc = functools.partial(negativevaluefunction, betaEVfunc, exogstatevec[s2])
        def currentposvalfunc(s1_new_val):
            return(-currentnegvalfunc(s1_new_val))
        # get s1new that maximises utility
        pstar = fminbound(currentnegvalfunc, endogstatevec[0], endogstatevec[-1])
        # Vstar
        Vstar = currentposvalfunc(pstar) - menu

        def currentposvalfunc_diff(s1_new_val):
            return(currentposvalfunc(s1_new_val) - Vstar)
            

        # get value from not changing
        Vnochange_low = currentposvalfunc_diff(endogstatevec[0] / (1 + inflation))
        Vnochange_high = currentposvalfunc_diff(endogstatevec[-1] / (1 + inflation))
        if pstar > endogstatevec[0] / (1 + inflation) and pstar < endogstatevec[-1] / (1 + inflation):
            # optimal pstar is within points I can consider
            if Vnochange_low < 0:
                # price at which agents switch from choosing to change price to not change price is within points I can consider
                # need to adjust for inflation since real price i.e. if inflation is +ve and don't change price then price was relatively higher yesterday
                priceboundlow = brentq(currentposvalfunc_diff, endogstatevec[0] / (1 + inflation), pstar) * (1 + inflation)
            else:
                # price at which agents switch not within points I can consider
                priceboundlow = endogstatevec[0]
            if Vnochange_high < 0:
                # price at which agents switch from choosing not to change price to change price is within points I can consider
                priceboundhigh = brentq(currentposvalfunc_diff, pstar, endogstatevec[-1] / (1 + inflation)) * (1 + inflation)
            else:
                # price at which agents switch not within points I can consider
                priceboundhigh = endogstatevec[-1]
        elif pstar <= endogstatevec[0] / (1 + inflation):
            # optimal pstar is lower than points I can consider
            if Vnochange_low <= 0:
                # price at which agents switch from choosing not to change price to change price is below range I can consider
                priceboundlow = endogstatevec[0]
                priceboundhigh = endogstatevec[0]
            elif Vnochange_high >= 0:
                # price at which agents switch from choosing not to change price to change price is above range I can consider
                priceboundlow = endogstatevec[0]
                priceboundhigh = endogstatevec[-1]
            elif Vnochange_low > 0 and Vnochange_high < 0:
                # price at which agents switch from choosing not to change price to change price is in range I can consider
                priceboundlow = endogstatevec[0]
                priceboundhigh = brentq(currentposvalfunc_diff, endogstatevec[0] / (1 + inflation), endogstatevec[-1] / (1 + inflation)) * (1 + inflation)
            else:
                raise ValueError('Should satisfy one of these conditions. Something wrong.')
        elif pstar >= endogstatevec[-1] / (1 + inflation):
            # optimal pstar is higher than points I can consider
            if Vnochange_high <= 0:
                # price at which agents switch from choosing to change price not to change price is above range I can consider
                priceboundlow = endogstatevec[-1]
                priceboundhigh = endogstatevec[-1]
            elif Vnochange_low >= 0:
                # price at which agents switch from choosing to change price not to change price is below range I can consider
                priceboundlow = endogstatevec[0]
                priceboundhigh = endogstatevec[-1]
            elif Vnochange_low < 0 and Vnochange_high > 0:
                # price at which agents switch from choosing to change price not to change price is in range I can consider
                priceboundlow = brentq(currentposvalfunc_diff, endogstatevec[0] / (1 + inflation), endogstatevec[-1] / (1 + inflation)) * (1 + inflation)
                priceboundhigh = endogstatevec[-1]
            else:
                raise ValueError('Should satisfy one of these conditions. Something wrong.')
        
        pol.append([pstar, priceboundlow, priceboundhigh])

    return(pol)


# Aggregating Policy Functions:{{{1
def getpolprobs(pol, inflation, pol_pi = None, num_pol_pi = 100):
    """
    Inputs: For a given productivity, optimal price is pstar and only change if not in range [pnochangelow, pnochangehigh]
    Use this to get the probability that a price, productivity pair will take a specific price value in the next period

    Method:
    Price [-0.1, -0.05, 0, 0.05, 0.1] represent [<-0.075, -0.075 < x < -0.025, -0.025 < x < 0.025, 0.025 < x < 0.075, x > 0.075]
    First find prob that range does not change price
    Assume for lowest and highest price that price always changes
    Then for given price, polprobs in next period are given by weighted sum of price changing and price not changing
    """

    ns2 = len(pol)

    if pol_pi is None:
        lowest = np.min([pol[s2][1] for s2 in range(ns2)])
        highest = np.max([pol[s2][2] for s2 in range(ns2)])
        pol_pi = np.linspace(lowest, highest, num_pol_pi)

    ns1 = len(pol_pi)

    midpoints = (np.array(pol_pi[1: ]) + np.array(pol_pi[: -1])) / 2

    polprobs = np.empty([ns1, ns2, ns1])
    price_change_prob = np.empty([ns1, ns2])


    for s2 in range(ns2):
        sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
        from dist_func import intervals_gt
        probabovelowbound = intervals_gt(pol[s2][1], midpoints)
        sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
        from dist_func import intervals_gt
        probabovehighbound = intervals_gt(pol[s2][2], midpoints)
        # probability that a firm with productivity state s2 will change over different previous price states s1
        probnochange = np.concatenate(([0], np.array(probabovehighbound) - np.array(probabovelowbound), [0]))

        # rewrite pstar as probability of discrete price states
        sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
        from dist_func import weightvaluediscretevec
        pstarvec = weightvaluediscretevec(pol[s2][0], pol_pi)

        for s1 in range(ns1):
            probnochange_thisone = probnochange[s1]
            sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
            from dist_func import weightvaluediscretevec
            pnochangevec = weightvaluediscretevec(pol_pi[s1] / (1 + inflation), pol_pi)

            polprobs[s1, s2] = probnochange_thisone * np.array(pnochangevec) + (1 - probnochange_thisone) * np.array(pstarvec)

            price_change_prob[s1, s2] = 1 - probnochange_thisone

    return(polprobs, price_change_prob, pol_pi)


def getpolstate(pol, inflation):
    """
    Get the price states that would be realised under pol.
    """
    if len(pol) > 1:
        raise ValueError('Not set up getpolstate for more than one productivity yet.')
    if inflation == 0:
        raise ValueError('getpolstate method does not work with zero inflation.')

    # initial state is pstar
    states = [pol[0][0]]
    while True:
        # stop if price set in last period was outside previous price no change bounds
        if states[-1] < pol[0][1] or states[-1] > pol[0][2]:
            break
        # otherwise add the relative price that will be yielded from not changing the previous price
        # note we need to adjust this by inflation
        states.append(states[-1] / (1 + inflation))

    # order numerically
    states = np.array(sorted(states))

    numstates = len(states)
    endogstatedist = np.repeat(1/numstates, numstates)
    prob_change_value = endogstatedist[0]

    # add fullstatedist - don't need for moment
    fullstatedist = None

    return(fullstatedist, endogstatedist, states, prob_change_value)


def getdist_continuous_menu(transmissionarray, polprobs, checks = False):

    if False:
        # get probability of moving from (s1, s2) to (s1', s2')
        sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
        from vfi_1endogstate_func import gentransmissionstararray_1endogstate_polprobs
        transmissionstararray = gentransmissionstararray_1endogstate_polprobs(transmissionarray, polprobs)
        
        ns1 = np.shape(polprobs)[0]
        sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
        from vfi_1endogstate_func import getstationarydist_1endogstate_full
        fullstatedist, endogstatedist = getstationarydist_1endogstate_full(transmissionstararray, ns1)

    else:
        sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
        from vfi_1endogstate_func import getstationarydist_1endogstate_direct
        fullstatedist, endogstatedist = getstationarydist_1endogstate_direct(transmissionarray, polprobs, crit = 1e-9)

    return(fullstatedist, endogstatedist)


def checkdist(endogstatedist):
    # get weighting on points close to the lower bound and upper bound to verify I don't need to raise the range of prices
    numpoints = len(endogstatedist)

    # get lowest 2.5 percentile of points
    lowpoints = [i for i in range(numpoints) if i / numpoints < 0.025]
    highpoints = [i for i in range(numpoints) if (numpoints - i - 1) / numpoints < 0.025]

    lowpointprob = np.sum([endogstatedist[i] for i in lowpoints])
    highpointprob = np.sum([endogstatedist[i] for i in highpoints])

    return(lowpointprob, highpointprob)



# Rest of Doc - Case where Exogenous = Productivity:{{{1
# Basic Setup:{{{1
def simpleprofitfunc_p(p, p_i, A_i):
    profits = p_i ** (1 - p['SIGMA']) - p['MC'] / A_i * p_i ** (-p['SIGMA'])

    return(profits)
        

def getparamssdict(p = None):

    if p is None:
        p = {}

    # IF WANT TO RUN QUICKLY - OTHERWISE COMMENT OUT
    if 'doquick' in p and p['doquick'] is True:
        p['num_pi'] = 100
        p['num_A'] = 10
        print('DOING QUICK IMPRECISE RESULTS VERSION')


    # END COMMENT OUT RUN QUICKLY


    p_defaults = {}
    # default parameters for distributions - taken from Nakamura Steinsson (2008)
    p_defaults['rho_A'] = 0.66 # matches Nakamura Steinsson 2008
    p_defaults['sd_Ashock'] = 0.0428 # matches Nakamura Steinsson 2008
    p_defaults['BETA'] = 0.96 # matches Nakamura Steinsson 2008

    p_defaults['SIGMA'] = SIGMA
    p_defaults['pistar'] = 0.02

    p_defaults['basicchecks'] = False
    p_defaults['printinfo'] = False
    p_defaults['printinfosummary'] = True
    p_defaults['continuousV'] = False
    p_defaults['returnpol'] = False
    p_defaults['pol_details'] = True
    for param in p_defaults:
        if param not in p:
            p[param] = p_defaults[param]

    # calibration menu cost:{{{
    menufile = __projectdir__ / Path('temp/calib_beta_sigma/') + str(p['BETA']) + '_' + str(p['SIGMA'])
    defaultmenu = 0.04157
    if 'menu' in p:
        None
    elif 'defaultmenu' in p and p['defaultmenu'] is True:
        p['menu'] = defaultmenu
    elif os.path.isfile(menufile):
        with open(menufile) as f:
            menu = f.read()
        if menu[-1] == '\n':
            menu = menu[: -1]
        p['menu'] = float(menu)
    else:
        print('Warning: Menu cost not specified in dict or in file. Using default.')
        p['menu'] = defaultmenu

    # calibration menu cost:}}}

    # adjust printing
    if p['printinfo'] is True:
        p['printinfosummary'] = True
    # add marginal costs
    if 'MC' not in p:
        p['MC'] = (p['SIGMA'] - 1) / p['SIGMA']

    # add adjusted BETA
    if 'BETA_a' not in p:
        p['BETA_a'] = p['BETA'] ** (1/12)
    # add alternative inflation measures
    if 'Pistar' not in p:
        p['Pistar'] = 1 + p['pistar']
    if 'Pistar_a' not in p:
        p['Pistar_a'] = p['Pistar'] ** (1/12)
    if 'pistar_a' not in p:
        p['pistar_a'] = p['Pistar_a'] - 1

    # add p_i
    if 'num_pi' not in p and 'logp_i' not in p:
        # raising num_pi doesn't seem to make much difference
        p['num_pi'] = 400
    if 'num_pol_pi' not in p:
        p['num_pol_pi'] = p['num_pi']

    if 'logp_i' not in p and 'p_i' not in p:
        p['logp_i'] = np.linspace(-0.15, 0.15, p['num_pi'])
    if 'p_i' not in p:
        p['p_i'] = np.exp(p['logp_i'])


    if 'Aval' not in p:
        # need to get Aval and transmissionarray

        if 'num_A' not in p:
            # raising num_A does seem to make a difference and raises MC and NU for a given iteration
            p['num_A'] = 100

        sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
        from dist_func import values_markov_normal
        logAval, p['transmissionarray'] = values_markov_normal(p['num_A'], rho = p['rho_A'], sdshock = p['sd_Ashock'])
        p['Aval'] = np.exp(logAval)

    else:
        p['num_A'] = len(p['Aval'])
    # add probabilities of getting Aval
    sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
    from markov_func import getstationarydist
    p['Aprobs'] = getstationarydist(p['transmissionarray'])


    if p['continuousV'] is False:
        p['profitarray'] = np.empty([p['num_pi'], len(p['Aval'])])
        for i in range(p['num_pi']):
            for j in range(len(p['Aval'])):
                p['profitarray'][i, j] = simpleprofitfunc_p(p, p['p_i'][i], p['Aval'][j])

    # need to define this after I define all parameters
    if 'profitfunc' not in p:
        p['profitfunc'] = functools.partial(simpleprofitfunc_p, p)

    return(p)


# Solve for Value Function:{{{1
def value_p(p):
    if p['printinfosummary'] is True:
        print('Start get value function: ' + str(datetime.datetime.now() - starttime))
    # don't bother solving for pol since have better method for doing it
    if p['continuousV'] is True:
        ret = vficontinuous(p['profitfunc'], p['p_i'], p['Aval'], p['transmissionarray'], p['BETA_a'], p['menu'], inflation = p['pistar_a'], crit = 1e-5, printinfo = p['printinfo'], basicchecks = p['basicchecks'], returnpol = p['returnpol'])
    else:
        ret = vfidiscrete(p['profitarray'], p['p_i'], p['transmissionarray'], p['BETA_a'], p['menu'], inflation = p['pistar_a'], printinfo = p['printinfo'], basicchecks = p['basicchecks'], returnpol = p['returnpol'])

    if p['returnpol'] is True:
        p['V'], p['pol'], p['price_change'] = ret
    else:
        p['V'] = ret

    return(p)


def value_p_test():
    p0 = {'num_pi': 50, 'BETA': 0.95, 'returnpol': True, 'basicchecks': True}
    p0 = getparamssdict(p0)
    p1 = copy.deepcopy(p0)
    p1['continuousV'] = True

    p0 = value_p(p0)

    p1 = value_p(p1)

    print('V:')
    print(p0['V'])
    print(p1['V'])
    print('\npol:')
    print(p0['pol'])
    print(p1['pol'])
    print('\nprice_change:')
    print(p0['price_change'])
    print(p1['price_change'])

    p = value_p(p0)

# Solve for Policy Function and Price Distribution:{{{1
def polrough1_p(p):
    """
    polsingleproductivity probably the way to go if I want to avoid discontinuities in general equilibrium.

    Just get the standard discrete policy function from the value function iteration.
    So I don't need to do value function first.
    """
    # need to return pol
    p['returnpol'] = True
    p = value_p(p)

    # prices that policy function relate to
    p['pol_pi'] = p['p_i']

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import getpolprobs_1endogstate_continuous
    p['polprobs'] = getpolprobs_1endogstate_continuous(p['pol'], p['pol_pi'])

    p['fullstatedist'], p['endogstatedist'] = getdist_continuous_menu(p['transmissionarray'], p['polprobs'])

    p['price_change_prob'] = np.sum(p['fullstatedist'] * p['price_change'])

    return(p)


def polrough2_p(p):
    """
    polsingleproductivity works much better

    This one works by smoothing the policy function you get out of the basic value function iteration.
    """
    p['pol'], p['price_change'] = getpol_quick(p['profitfunc'], p['p_i'], p['Aval'], p['transmissionarray'], p['BETA_a'], p['menu'], p['pistar_a'], p['V'])

    # prices that policy function relate to
    p['pol_pi'] = p['p_i']

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import getpolprobs_1endogstate_continuous
    p['polprobs'] = getpolprobs_1endogstate_continuous(p['pol'], p['pol_pi'])

    p['fullstatedist'], p['endogstatedist'] = getdist_continuous_menu(p['transmissionarray'], p['polprobs'])

    p['price_change_prob'] = np.sum(p['fullstatedist'] * p['price_change'])

    return(p)


def polmultipleproductivities_p(p):
    """
    This only works when I have multiple productivity states.
    Need multiple otherwise if I have initial mass on a given price, all firms will do the same and it will take a long time to converge.

    Solve for pol properly - need to already have computed the value function.
    Then solve for distributions and probability of price changes.
    """
    # solve for the policy function
    if p['printinfosummary'] is True:
        print('Start get policy function: ' + str(datetime.datetime.now() - starttime))
    p['pol'] = getpol(p['profitfunc'], p['p_i'], p['Aval'], p['transmissionarray'], p['BETA_a'], p['menu'], p['pistar_a'], p['V'])

    if p['printinfosummary'] is True:
        print('Start get policy probabilities: ' + str(datetime.datetime.now() - starttime))
    # p['polprobs'], p['price_change'], p['pol_pi'] = getpolprobs(p['pol'], p['pistar_a'], pol_pi = p['p_i'])
    p['polprobs'], p['price_change'], p['pol_pi'] = getpolprobs(p['pol'], p['pistar_a'], num_pol_pi = p['num_pol_pi'])

    if p['printinfosummary'] is True:
        print('Start get distributions: ' + str(datetime.datetime.now() - starttime))
    p['fullstatedist'], p['endogstatedist'] = getdist_continuous_menu(p['transmissionarray'], p['polprobs'])



    if p['pol_details'] is True:
        # save snapshops of the policy function
        p['polfirst'] = p['pol'][0]
        p['pollast'] = p['pol'][-1]
        p['polmedian'] = p['pol'][len(p['pol']) // 2]

        p['price_change_prob'] = np.sum(p['fullstatedist'] * p['price_change'])

        nump = len(p['pol_pi'])
        numA = len(p['Aval'])


        # get change in price matrix
        pstarmat = np.empty([nump, numA])
        plowmat = np.empty([nump, numA])
        phighmat = np.empty([nump, numA])
        for j in range(numA):
            pstarmat[:, j] = p['pol'][j][0]
            plowmat[:, j] = p['pol'][j][1]
            phighmat[:, j] = p['pol'][j][2]
        pcurmat = np.repeat(p['pol_pi'] / p['Pistar_a'], numA).reshape([nump, numA])

        # size of price change
        logpstarchangemat = np.log(pstarmat) - np.log(pcurmat)
        logpstarposchangemat = np.maximum(logpstarchangemat, 0)
        logpstarnegchangemat = np.maximum(-logpstarchangemat, 0)

        # indicator matrix for positive or negative
        pospricechangemat = logpstarposchangemat > 0
        negpricechangemat = logpstarnegchangemat > 0

        # indicator for (p_i, a_i) pairs where price went up/down
        posind = pospricechangemat * p['price_change']
        negind = negpricechangemat * p['price_change']
        
        # probability price goes up/down
        p['price_change_prob_pos'] = np.sum(posind * p['fullstatedist'])
        p['price_change_prob_neg'] = np.sum(negind * p['fullstatedist'])

        # size by which goes up/down
        if p['price_change_prob_pos'] != 0:
            p['price_change_size_pos'] = np.sum(logpstarposchangemat * p['fullstatedist'] * p['price_change']) / p['price_change_prob_pos']
        else:
            p['price_change_size_pos'] = np.nan
        if p['price_change_prob_neg'] != 0:
            p['price_change_size_neg'] = np.sum(logpstarnegchangemat * p['fullstatedist'] * p['price_change']) / p['price_change_prob_neg']
        else:
            p['price_change_size_neg'] = np.nan

        # print((logpstarnegchangemat * negpricechangemat).tolist())

        p['price_change_size_abs'] = p['price_change_size_pos'] * p['price_change_prob_pos'] / p['price_change_prob'] + p['price_change_size_neg'] * p['price_change_prob_neg'] / p['price_change_prob']

    return(p)


def polsingleproductivity(p):
    """
    Solve for exact price states that firms will set
    Only works when there is a single productivity state

    Solve for pol properly - so I actually compute the range over which firms change and do not change their price (rather than an imprecise range)
    Need to already have computed the value function.
    Then solve for distributions and probability of price changes.
    """
    # solve for the policy function
    p['pol'] = getpol(p['profitfunc'], p['p_i'], p['Aval'], p['transmissionarray'], p['BETA_a'], p['menu'], p['pistar_a'], p['V'])

    p['fullstatedist'], p['endogstatedist'], p['pol_pi'], p['price_change_prob'] = getpolstate(p['pol'], p['pistar_a'])

    return(p)


def addaggdetails(p):
    """
    Add aggregate variables to dictionary.
    """
    p['cesrelprice_nopower'] = np.sum(p['endogstatedist'] * p['pol_pi'] ** (1 - p['SIGMA']))
    p['cesrelprice'] = p['cesrelprice_nopower'] ** (1 / (1 - p['SIGMA']))

    # price dispersion
    # p['NUold'] = np.sum(p['endogstatedist'] * p['pol_pi'] ** (- p['SIGMA']))
    pricepart = np.tile(p['pol_pi'] ** (-p['SIGMA']), [p['num_A'], 1]).transpose()
    productivitypart = np.tile(1 / p['Aval'], [p['num_pol_pi'], 1])
    p['NU'] = np.sum(p['fullstatedist'] * pricepart * productivitypart)

    p['aggMC'] = p['MC'] * p['NU']
    p['profitshare'] = 1 - p['MC'] * p['NU']
    if p['pol_details'] is True:
        p['menushare'] = p['menu'] * p['price_change_prob']
        p['profitsharemenu'] = 1 - p['MC'] * p['NU'] - p['menushare']
    return(p)
    

def pol_test():
    """
    Test different methods for solving for the policy function.

    polrough_2 does not work well

    polrough1_p, polmultipleproductivities_p not work well for only one A since convergence takes a very long time as I have to wait for the dist vector to converge to a uniform distribution (which only occurs because of the approximations involved in the policy function)
    Need low number of states here (or could use alternative method for solving for transmissionstararray)
    """
    p0 = {'num_pi': 100, 'BETA': 0.1, 'returnpol': True, 'basicchecks': True, 'pistar': 0.01, 'continuousV': True, 'printinfosummary': False}
    print('get value function')
    p0 = getparamssdict(p0)
    p0v = value_p(p0)

    print('rough 1')
    pr1 = copy.deepcopy(p0)
    polrough1_p(pr1)
    pr1 = addaggdetails(pr1)

    print('rough 2')
    pr2 = copy.deepcopy(p0v)
    polrough2_p(pr2)
    pr2 = addaggdetails(pr2)

    print('proper with stateprobs')
    pp2 = copy.deepcopy(p0v)
    polmultipleproductivities_p(pp2)
    pp2 = addaggdetails(pp2)

    print('proper with state exact (only works with 1 exog state)')
    pp1 = copy.deepcopy(p0v)
    polsingleproductivity(pp1)
    pp1 = addaggdetails(pp1)

    print('State Distributions:')
    # get relevant states:{{{
    lowp = pp1['pol_pi'][0]
    highp = pp1['pol_pi'][-1]
    def lowpihighpi(pol_pi):
        """
        Get elements of pol_pi that bound lowp and highp.
        """
        for lowpi in range(len(pol_pi)):
            if pol_pi[lowpi + 1] > lowp:
                break
            lowpi += 1
        for highpi in reversed(list(range(len(pol_pi)))):
            if pol_pi[highpi - 1] < highp:
                break
            highpi -= 1
        return(lowpi, highpi)
    # }}}
    lowpi, highpi = lowpihighpi(pr1['pol_pi'])
    print(pr1['endogstatedist'][lowpi: highpi])
    lowpi, highpi = lowpihighpi(pr2['pol_pi'])
    print(pr2['endogstatedist'][lowpi: highpi])
    lowpi, highpi = lowpihighpi(pp2['pol_pi'])
    print(pp2['endogstatedist'][lowpi: highpi])
    print(pp1['endogstatedist'])

    print('Price Change Prob:')
    print(pr1['price_change_prob'])
    print(pr2['price_change_prob'])
    print(pp2['price_change_prob'])
    print(pp1['price_change_prob'])

    print('\nCES Price Sum:')
    print(pr1['cesrelprice'])
    print(pr2['cesrelprice'])
    print(pp2['cesrelprice'])
    print(pp1['cesrelprice'])

# Partial Equilibrium Solve:{{{1
def partialeq_solve_p(p):
    """
    Solve for given MC.
    """
    # add dictionary
    p = getparamssdict(p)

    # in case multiple iterations, separate this print from the previous print
    if p['printinfosummary'] is True:
        print('\n')

    # solve for the value function
    p = value_p(p)

    if len(p['Aval']) > 1:
        polmultipleproductivities_p(p)
    else:
        if len(p['p_i']) > 300:
            polrough1_p(p)
        else:
            polsingleproductivity(p)

    # check dist makes sense
    p['lowpointprob'], p['highpointprob'] = checkdist(p['endogstatedist'])

    # add aggregate details
    addaggdetails(p)


    if p['printinfosummary'] is True:
        toprint = sorted([name + ': ' + str(p[name]) for name in keydetails if name in p])
        for item in toprint:
            print(item)

    return(p)


# partialeq_solve_p({'pistar': 0.02, 'MC': 0.8818, 'num_A': 50, 'num_pi': 200, 'menu': 0.1})
def partialeq_solve_p_test():
    """
    Note that the menu cost is specified here.
    """
    p = {'pistar': 0.011, 'num_A': 2, 'num_pi': 3, 'printinfosummary': False}
    p = partialeq_solve_p(p = p)
    print(p['pistar'])
    print(p['cesrelprice'])


def partialeq_solve_p_test_sizeA():
    """
    This function makes the point that I need to get a large number of A to get smooth results.

    With higher inflation, the prices set by firms should fall with means \int_0^1p_i^{1 - sigma} should rise

    However, with a low number of A this doesn't always hold
    The reason is that COMPLETE
    """
    print('numA: ' + str(10) + '.')
    # inflation of 0.1
    p = {'pistar': 0.01, 'num_A': 10, 'printinfosummary': False}
    p = partialeq_solve_p(p = p)
    print(p['cesrelprice'])

    # inflation of 0.11
    p = {'pistar': 0.011, 'num_A': 10, 'printinfosummary': False}
    p = partialeq_solve_p(p = p)
    print(p['cesrelprice'])

    print('numA: ' + str(100) + '.')
    # inflation of 0.1
    p = {'pistar': 0.01, 'num_A': 100, 'printinfosummary': False}
    p = partialeq_solve_p(p = p)
    print(p['cesrelprice'])

    # inflation of 0.11
    p = {'pistar': 0.011, 'num_A': 100, 'printinfosummary': False}
    p = partialeq_solve_p(p = p)
    print(p['cesrelprice'])


# General Equilibrium Solve:{{{1
def generaleq_aux_p(p, MC):
    """
    Auxilliary function used when do generaleq_solve
    """
    # need to make copy every time I run function to ensure I don't end up running on same p
    p = copy.deepcopy(p)
    if p is None:
        p = {}
    p['MC'] = MC
    # no point in solving for policy function details
    p['pol_details'] = False
    partialeq_solve_p(p)
    if p['printinfosummary'] is True:
        print('\nMC solve current iteration cesrelprice: ' + str(p['cesrelprice']) + '. Time: ' + str(datetime.datetime.now() - starttime) + '.')

    return(p['cesrelprice'] - 1)


def fullsolve_p(p, MClow = None, MChigh = None, tolerance = 1e-6):
    """
    Fully solve for p.
    Solve for MC in general equilibrium if MC not specified. Otherwise, solve partial equilibrium.
    """
    if p is None:
        p = {}

    if 'MC' not in p:
        # want a small range to prevent errors and make code run faster
        # unless I have num_A very small or extreme values, I should always be in this approximate range
        if 'SIGMA' not in p:
            p['SIGMA'] = SIGMA
        if MClow is None:
            MClow = (p['SIGMA'] - 1) / p['SIGMA'] - 0.025
        if MChigh is None:
            MChigh = (p['SIGMA'] - 1) / p['SIGMA'] + 0.025

        f1 = functools.partial(generaleq_aux_p, p)

        p['MC'] = brentq(f1, MClow, MChigh, xtol = tolerance)

    # get pol details
    p['pol_details'] = True
    # add rest of solution
    p = partialeq_solve_p(p)

    if p['printinfosummary'] is True:
        print('\n\nSOLUTION FOR MC: ' + str(p['MC']) + '. Time: ' + str(datetime.datetime.now() - starttime))

    return(p)


def fullsolve_p_test():
    """
    Since we have no menu costs, this should equal approximately 0.75 regardless of the value of pistar.
    """
    p = {'num_A': 1, 'menu': 0.0, 'pistar': 0.1, 'num_pi': 1000, 'printinfosummary': False}
    p = generaleq_solve_p(p = p)
    print('MC:')
    print(p['MC'])


# Calibration General Functions:{{{1
def solvemenu_aux_givenMC(p, pricechangeprob, menu):
    p['menu'] = menu
    partialeq_solve_p(p)
    if p['printinfosummary'] is True:
        print('\nMenu solve current iteration of price_change_prob diff:' + str(p['price_change_prob'] - pricechangeprob) + '. Time: ' + str(datetime.datetime.now() - starttime) + '.')
    return(p['price_change_prob'] - pricechangeprob)


def solvemenu_givenMC(p = None, pricechangeprob = 0.087, inflation = 0.02562, menulow = 0.001, menuhigh = 0.1, tolerance = 1e-6):
    """
    Price frequency probability of 0.087 is taken from NS (2008)
    This was recorded using 1998-2005 data when CPI rose from 161.8 (Dec 1997) to 198.1 (Dec 2005) implying an annual inflation rate of 2.562%

    Rate of inflation important since under higher inflation firms change their price more frequently meaning menu costs must be larger to obtain a given price_change_prob


    IMPORTANT: This solves for the partial equilibrium level of menu costs. It does not solve for the model under the correct level of MC. I find that when I raise MC, price change prob goes down so the solution for the menu cost goes down (though not by much).
    """
    if p is None:
        p = {}
    p['pistar'] = inflation
    f1 = functools.partial(solvemenu_aux_givenMC, p, pricechangeprob)
    menustar = brentq(f1, menulow, menuhigh, xtol = tolerance)

    if p['printinfosummary'] is True:
        print('\n\nSOLUTION FOR MENU: ' + str(menustar))
    return(menustar)


def solvemenu_full(p0 = None, pricechangeprob = 0.087, inflation = 0.02562, printthis = True, menulow_input = 0.03, menuhigh_input = 0.05, menuprecision = 2e-4, MCprecision = 2e-4, raiseerrorifexitearly = False):
    """
    solvemenu_givenMC does solvemenu(MC) = menu* where MC is not MC*
    I want to iterate over solveMC(menu) and solvemenu(MC) to find menu*, MC* s.t. solveMC(menu*) = MC* and solvemenu(MC*) = menu*

    To do this I first solve for MClow and MChigh given menulow and menuhigh under MCinit
    I then do the same for menulow and menuhigh
    I keep doing this until I hopefully converge

    Precision shouldn't be too narrow since at small values I think the solve functions could go in the wrong direction.
    Normal directions:
    - When MC rises, solved for menu* goes down. I think this is because for a given menu, with higher MC, firms get lower benefit from changing prices so price change probabilities are lower. Therefore, we need a lower menu cost to get higher price change probabilities.
    - When menu rises, MC goes down. I think this is because with a higher menu, price dispersion is lower so the CES price aggregator is naturally lower. Therefore, MC does not need to rise by as much.
    """
    if p0 is None:
        p0 = {}
    p0['pistar'] = inflation
    # if 'SIGMA' not in p0:
    #     p0['SIGMA'] = SIGMA
    # if MCinit is None:
    #     MCinit = (p0['SIGMA'] - 1) / p0['SIGMA']

    solveformenu_thisiteration = False
    iterationi = 0
    # initial range for MC
    MClow_input = 0
    MChigh_input = 1
    menulow_output = menulow_input
    menuhigh_output = menuhigh_input
    while True:
        # do actual iteration
        if solveformenu_thisiteration is True:
            p = copy.deepcopy(p0)
            p['MC'] = MClow_input
            # constrain to consider only menu costs between menulow_input and menuhigh_input
            # I solve for tolerance = menuprecision / 5 since this is a different precision to the difference between MClow and MChigh
            # I need a smaller precision than that difference to avoid the possibility of no convergence
            menuhigh_output = solvemenu_givenMC(p = p, menulow = menulow_input, menuhigh = menuhigh_input, tolerance = menuprecision / 5)

            p = copy.deepcopy(p0)
            p['MC'] = MChigh_input
            menulow_output = solvemenu_givenMC(p = p, menulow = menulow_input, menuhigh = menuhigh_input, tolerance = menuprecision / 5)

        else:
            p = copy.deepcopy(p0)
            p['menu'] = menulow_input
            p = fullsolve_p(p, MClow = MClow_input, MChigh = MChigh_input, tolerance = MCprecision / 5)
            MChigh_output = p['MC']

            p = copy.deepcopy(p0)
            p['menu'] = menuhigh_input
            p = fullsolve_p(p, MClow = MClow_input, MChigh = MChigh_input, tolerance = MCprecision / 5)
            MClow_output = p['MC']

        # print basic details on iterations
        if p['printinfosummary'] is True:
            print('\n\n\nITERATION COMPLETED: ' + str(iterationi))
            print('Iteration was solving for menu: ' + str(solveformenu_thisiteration))
            print('menulow: ' + str(menulow_output))
            print('menuhigh: ' + str(menuhigh_output))
            print('MClow: ' + str(MClow_output))
            print('MChigh: ' + str(MChigh_output))

        # checks to ensure iterations haven't failed
        if MClow_output >= MChigh_output or MClow_output < MClow_input or MChigh_output > MChigh_input or menulow_output >= menuhigh_output or menulow_output < menulow_input or menuhigh_output > menuhigh_input:
            if raiseerrorifexitearly is True:
                raise ValueError('Iteration Failed.')
            else:
                print('WARNING: Iteration Failed.')
                print('Best guess for menu cost: ' + str(0.5 * (menulow_output + menuhigh_output)))
                return(0.5 * (menulow_output + menuhigh_output))

        if MChigh_output - MClow_output < MCprecision and menuhigh_output - menulow_output < menuprecision:
            print('Menu cost solution: ' + str(0.5 * (menulow_output + menuhigh_output)))
            return(0.5 * (menulow_output + menuhigh_output))

        # solve for other one next iteration
        if solveformenu_thisiteration is True:
            menuhigh_input = menuhigh_output
            menulow_input = menulow_output
        else:
            MChigh_input = MChigh_output
            MClow_input = MClow_output
        solveformenu_thisiteration = not solveformenu_thisiteration
        iterationi += 1


def solvemenu_beta_sigma_aux(savefolder, p0, menulow, menuhigh, betasigmatuple, printinfosummary = False):
    beta = betasigmatuple[0]
    sigma = betasigmatuple[1]
    print('Started: solve for menu cost. Beta: ' + str(beta) + '. Sigma: ' + str(sigma) + '. Time: ' + str(datetime.datetime.now() - starttime) + '.')

    p = copy.deepcopy(p0)
    p['BETA'] = beta
    p['SIGMA'] = sigma
    p['printinfosummary'] = printinfosummary

    if menulow is None:
        # I know the case for sigma == 8 is between 0.03 - 0.05
        # otherwise set wide bound
        if sigma == 8:
            menulow = 0.03
        else:
            menulow = 0.001
    if menuhigh is None:
        # I know the case for sigma == 8 is between 0.03 - 0.05
        # otherwise set wide bound
        if sigma == 8:
            menuhigh = 0.05
        else:
            menuhigh = 0.15
    
    menustar = solvemenu_full(p, menulow_input = menulow, menuhigh_input = menuhigh)
    with open(savefolder + str(beta) + '_' + str(sigma), 'w+') as f:
        f.write(str(menustar))

    print('Finished: solve for menu cost. Beta: ' + str(beta) + '. Sigma: ' + str(sigma) + '. Time: ' + str(datetime.datetime.now() - starttime) + '.')


def solvemenu_betas_sigmas(savefolder = __projectdir__ / Path('temp/calib_beta_sigma/'), p = {}, multiprocessthis = False, menulow = None, menuhigh = None):
    """
    Allow for a wide range of menu costs given that I consider different BETAs and SIGMAs
    """
    if os.path.isdir(savefolder):
        shutil.rmtree(savefolder)
    os.mkdir(savefolder)

    # get tuples:{{{
    tuples = []
    # basic calibration
    betadef = 0.96
    sigmadef = 8
    tuples.append([betadef, sigmadef])

    # different betas
    betas = betas_list_default
    # skip 0.96 since already done
    if betadef in betas:
        betas.remove(betadef)
    for beta in betas:
        tuples.append([beta, sigmadef])

    # different sigmas
    sigmas = sigmas_list_default
    # skip 8 since already done
    if sigmadef in sigmas:
        sigmas.remove(sigmadef)
    for sigma in sigmas:
        tuples.append([betadef, sigma])
    # get tuples:}}}

    f1 = functools.partial(solvemenu_beta_sigma_aux, savefolder, p, menulow, menuhigh)

    if multiprocessthis is True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(f1, tuples)
    else:
        for tuple1 in tuples:
            f1(tuple1, printinfosummary = True)


def solvemenu_betas_sigmas_test(multiprocessthis = False):
    """
    Verifies that solvemenu_betas_sigmas works ok.
    This can fail due to lack of precision (since I dramatically reduce the state spaces).
    """
    p = {'doquick': True}
    solvemenu_betas_sigmas(savefolder = __projectdir__ / Path('temp/calib_beta_sigma_temp/'), p = p, multiprocessthis = multiprocessthis)


# Save Function General:{{{1
def savesinglerun(filenamenostem, p, skipfileifexists = False):
    if skipfileifexists is True and os.path.isfile(filenamenostem + '.pickle') is True:
        print('Skipped file: ' + filenamenostem + '.')
        return(0)
            
    fullsolve_p(p)

    savedict = {name: p[name] for name in keydetails if name in p}
    with open(filenamenostem + '.pickle', 'wb') as f:
        pickle.dump(savedict, f)

    toprint = sorted([name + ': ' + str(savedict[name]) for name in savedict])
    with open(filenamenostem + '.txt', 'w+') as f:
        f.write('\n'.join(toprint))

    

def savesingleparamfolder_aux(p0, folder, paramname, skipfileifexists, param):
    p = copy.deepcopy(p0)
    p[paramname] = param

    print('Started: ' + paramname + ': ' + str(param) + '. Time: ' + str(datetime.datetime.now() - starttime) + '.')

    savesinglerun(folder + str(param), p, skipfileifexists = skipfileifexists)

    print('Finished: ' + paramname + ': ' + str(param) + '. Time: ' + str(datetime.datetime.now() - starttime) + '.')


def savesingleparamfolder(folder, paramname, params, p0 = None, replacefolder = True, skipfileifexists = False, multiprocessthis = False):
    """
    Should input params as floats/integers
    Should not use linspace to generate these params since then I can get 0.199999 rather than 0.2

    Runs fullsolve for a list of parameters.
    By default, it uses the standard dictionary.
    """
    if p0 is None:
        p0 = {}
    if multiprocessthis is True:
        # would be very messy if I had multiprocessing with printinfosummary on
        p0['printinfosummary'] = False

    if replacefolder is True and os.path.isdir(folder):
        shutil.rmtree(folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # get inputlist for multiprocessing
    inputlist = []
    for i in range(len(params)):
        inputlist.append(params[i])

    # get auxilliary function
    f1 = functools.partial(savesingleparamfolder_aux, p0, folder, paramname, skipfileifexists)

    if multiprocessthis is True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(f1, inputlist)
    else:
        for i in range(len(inputlist)):
            f1(inputlist[i])


def savedoubleparamfolder_aux(p0, folder, firstparamname, secondparamname, skipfileifexists, thisinput):
    firstparam = thisinput[0]
    secondparam = thisinput[1]
    p = copy.deepcopy(p0)
    p[firstparamname] = firstparam
    p[secondparamname] = secondparam

    print('Started: ' + firstparamname + ': ' + str(firstparam) + '. ' + secondparamname + ': ' + str(secondparam) +  '. Time: ' + str(datetime.datetime.now() - starttime) + '.')

    savesinglerun(folder + str(firstparam) + '_' + str(secondparam), p, skipfileifexists = skipfileifexists)

    print('Finished: ' + firstparamname + ': ' + str(firstparam) + '. ' + secondparamname + ': ' + str(secondparam) +  '. Time: ' + str(datetime.datetime.now() - starttime) + '.')


def savedoubleparamfolder(folder, firstparamname, firstparams, secondparamname, secondparams, p0 = None, replacefolder = True, skipfileifexists = False, multiprocessthis = False):
    """
    Runs fullsolve for two sets of parameters
    If run for firstparam = 0.02 and secondparam = 0.03 then saves as 0.02_0.03
    Should input params as floats/integers
    """
    if p0 is None:
        p0 = {}
    if multiprocessthis is True:
        # would be very messy if I had multiprocessing with printinfosummary on
        p0['printinfosummary'] = False

    if replacefolder is True and os.path.isdir(folder):
        shutil.rmtree(folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # get inputlist for multiprocessing
    inputlist = []
    for i in range(len(firstparams)):
        for j in range(len(secondparams)):
            inputlist.append([firstparams[i], secondparams[j]])

    # get auxilliary function
    f1 = functools.partial(savedoubleparamfolder_aux, p0, folder, firstparamname, secondparamname, skipfileifexists)

    if multiprocessthis is True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(f1, inputlist)
    else:
        for i in range(len(inputlist)):
            f1(inputlist[i])


    
# Save Dicts:{{{1
def savepistars(multiprocessthis = False, replacefolder = True, skipfileifexists = False):
    # do it in this order to get a sense of what the results will be sooner
    pistars = [0.0, 0.02, 0.04, -0.01, -0.005, -0.001, 0.001, 0.005, 0.01, 0.015, 0.025, 0.03, 0.035, 0.06, 0.08, 0.1]
    savesingleparamfolder(__projectdir__ / Path('temp/pistars/'), 'pistar', pistars, replacefolder = replacefolder, skipfileifexists = skipfileifexists, multiprocessthis = multiprocessthis)


def savebetas(multiprocessthis = False, replacefolder = True, skipfileifexists = False):
    """
    Do high betas first since they take longer (makes sense with multiprocessing)
    """
    # betas = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    p = {'defaultmenu': True}
    betas = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9]
    savedoubleparamfolder(__projectdir__ / Path('temp/betas/'), 'pistar', [0, 0.02], 'BETA', betas, replacefolder = replacefolder, skipfileifexists = skipfileifexists, multiprocessthis = multiprocessthis, p0 = p)


def savesigmas(multiprocessthis = False, replacefolder = True, skipfileifexists = False):
    p = {'defaultmenu': True}
    savedoubleparamfolder(__projectdir__ / Path('temp/sigmas/'), 'pistar', [0, 0.02], 'SIGMA', [2, 3, 4, 5, 6, 7, 8, 9, 10], replacefolder = replacefolder, skipfileifexists = skipfileifexists, multiprocessthis = multiprocessthis, p0 = p)


def savemenus(multiprocessthis = False, replacefolder = True, skipfileifexists = False):
    savedoubleparamfolder(__projectdir__ / Path('temp/menus/'), 'pistar', [0, 0.02], 'menu', [0.03, 0.0325, 0.035, 0.0375, 0.04, 0.0425, 0.045, 0.0475, 0.05], replacefolder = replacefolder, skipfileifexists = skipfileifexists, multiprocessthis = multiprocessthis)


def savebetas_adjustedmenu(multiprocessthis = False, replacefolder = True, skipfileifexists = False):
    betas_string = betas_list_default
    savedoubleparamfolder(__projectdir__ / Path('temp/betas_adjustedmenu/'), 'pistar', [0, 0.02], 'BETA', betas_string, replacefolder = replacefolder, skipfileifexists = skipfileifexists, multiprocessthis = multiprocessthis)


def savesigmas_adjustedmenu(multiprocessthis = False, replacefolder = True, skipfileifexists = False):
    sigmas_string = sigmas_list_default
    savedoubleparamfolder(__projectdir__ / Path('temp/sigmas_adjustedmenu/'), 'pistar', [0, 0.02], 'SIGMA', sigmas_string, replacefolder = replacefolder, skipfileifexists = skipfileifexists, multiprocessthis = multiprocessthis)


def fullsavedicts(multiprocessthis = False, replacefolder = True, skipfileifexists = False):
    """ 
    multiprocessing seemed to be slower - not sure why - so I turned it off
    """
    # if os.path.isdir(__projectdir__ / Path('temp/')):
    #     shutil.rmtree(__projectdir__ / Path('temp/'))
    # os.mkdir(__projectdir__ / Path('temp/'))

    solvemenu_betas_sigmas(multiprocessthis = multiprocessthis)
    savepistars(multiprocessthis = multiprocessthis, replacefolder = replacefolder, skipfileifexists = skipfileifexists)
    savemenus(multiprocessthis = multiprocessthis, replacefolder = replacefolder, skipfileifexists = skipfileifexists)
    savebetas_adjustedmenu(multiprocessthis = multiprocessthis, replacefolder = replacefolder, skipfileifexists = skipfileifexists)
    savesigmas_adjustedmenu(multiprocessthis = multiprocessthis, replacefolder = replacefolder, skipfileifexists = skipfileifexists)

# Print Dicts:{{{1
def printfolderdict(folder, params):
    if not isinstance(params, list):
        params = [params]

    filenames = [filename for filename in os.listdir(folder) if filename.endswith('.pickle')]

    for param in params:
        for filename in sorted(filenames):
            with open(os.path.join(folder, filename), 'rb') as f:
                p = pickle.load(f)
            print('Filename: ' + filename[: -len('.pickle')] + '. ' + param + ': ' + str(p[param]) + '.')

# printfolderdict(__projectdir__ / Path('temp/menus/'), 'aggMC')
# Load Data From Save Dicts:{{{1
def loadsingleparamfolder(folder):
    paramstrs = [filename[: -len('.pickle')] for filename in os.listdir(folder) if filename.endswith('.pickle')]
    paramdict = {float(paramstr): paramstr for paramstr in paramstrs}

    params = sorted(paramdict)

    singlelist = []
    for param in params:
        paramstr = paramdict[param]

        with open(os.path.join(folder, paramstr + '.pickle'), 'rb') as f:
            singlelist.append(pickle.load(f))

    return(params, singlelist)


def loaddoubleparamfolder(folder):
    filenames = [filename for filename in os.listdir(folder) if filename.endswith('.pickle')]
    param1strs = set([filename.split('_')[0] for filename in filenames])
    param2strs = set([filename[: -len('.pickle')].split('_')[1] for filename in filenames])
    param1dict = {float(param1str): param1str for param1str in param1strs}
    param2dict = {float(param2str): param2str for param2str in param2strs}

    params1 = sorted(param1dict)
    params2 = sorted(param2dict)

    retlist = []
    for param1 in params1:
        param1str = param1dict[param1]
        retlist2 = []
        for param2 in params2:
            param2str = param2dict[param2]

            with open(os.path.join(folder, param1str + '_' + param2str + '.pickle'), 'rb') as f:
                retlist2.append(pickle.load(f))

        retlist.append(retlist2)

    return(params1, params2, retlist)


def getchangeprofitshare(folder, lowpistari = 0, highpistari = 1):
    """
    Results should be saved where pistar is the first parameter
    In percentage terms
    """
    pistars, secondparams, retlist = loaddoubleparamfolder(folder)
    lowpistarlist = retlist[lowpistari]
    highpistarlist = retlist[highpistari]
    changeprofitshares = [100 * ((1 - highpistarlist[i]['aggMC']) - (1 - lowpistarlist[i]['aggMC'])) for i in range(len(lowpistarlist))]

    return(secondparams, changeprofitshares)

# secondparams, changeprofitshares = getchangeprofitshare(__projectdir__ / Path('temp/menus/'))
# print(secondparams)
# print(changeprofitshares)
def interpolatepistar(pistar):
    """
    Read pistars from folder and then interpolate them to get estimates of MC, NU and menushare in that case
    pistar should be in range of pistars in folder
    """
    pistars, retdictlist = loadsingleparamfolder(__projectdir__ / Path('temp/pistars/'))

    MClist = [retdict['MC'] for retdict in retdictlist]
    NUlist = [retdict['NU'] for retdict in retdictlist]
    menusharelist = [retdict['menushare'] for retdict in retdictlist]
    
    MC = np.interp(pistar, pistars, MClist)
    NU = np.interp(pistar, pistars, NUlist)
    menushare = np.interp(pistar, pistars, menusharelist)

    return(MC, NU, menushare)

# Summary Graphs:{{{1
def graph_pistar_profitshare(show = False):
    pistars, plist = loadsingleparamfolder(__projectdir__ / Path('temp/pistars/'))
    profitshares = [100 * (1 - p['aggMC']) for p in plist]
    profitsharesmenu = [100 * (1 - p['aggMC'] - p['menushare']) for p in plist]

    plt.plot(pistars, profitsharesmenu, label = 'External Menu Costs')
    plt.plot(pistars, profitshares, label = 'Internal Menu Costs')

    plt.xlabel('Trend Inflation')
    plt.ylabel('Profit Share (\%)')
    
    plt.legend()

    plt.savefig(__projectdir__ / Path('temp/graphs/pistar_profitshare.png'))
    if show is True:
        plt.show()

    plt.clf()


def graph_pistar_pricechangeprob(show = False):
    pistars, plist = loadsingleparamfolder(__projectdir__ / Path('temp/pistars/'))
    pricechangeprobs = [p['price_change_prob'] for p in plist]

    plt.plot(pistars, pricechangeprobs)

    plt.xlabel('Trend Inflation')
    plt.ylabel('Price Change Probability')

    plt.savefig(__projectdir__ / Path('temp/graphs/pistar_pricechangeprob.png'))
    if show is True:
        plt.show()

    plt.clf()
    

def graph_pistar_pricechangesize(show = False):
    pistars, plist = loadsingleparamfolder(__projectdir__ / Path('temp/pistars/'))
    pricechangesizes = [p['price_change_size_abs'] for p in plist]

    plt.plot(pistars, pricechangesizes)

    plt.xlabel('Trend Inflation')
    plt.ylabel('Absolute Price Change Size')

    plt.savefig(__projectdir__ / Path('temp/graphs/pistar_pricechangesize.png'))
    if show is True:
        plt.show()

    plt.clf()


def graph_pistar_inactionsize(show = False):
    pistars, plist = loadsingleparamfolder(__projectdir__ / Path('temp/pistars/'))
    medianresetprice = [p['polmedian'][0] for p in plist]
    medianlb = [p['polmedian'][1] for p in plist]
    medianub = [p['polmedian'][2] for p in plist]

    plt.plot(pistars, medianresetprice, label = 'Median Reset Price')
    plt.plot(pistars, medianlb, label = 'Median Lower Bound')
    plt.plot(pistars, medianub, label = 'Median Upper Bound')

    plt.legend()

    plt.xlabel('Trend Inflation')
    # plt.ylabel('Absolute Price Change Size')

    plt.savefig(__projectdir__ / Path('temp/graphs/pistar_inactionsize.png'))
    if show is True:
        plt.show()

    plt.clf()


def graph_beta_adjustedmenu_changeprofitshare(show = False):
    betas, changeprofitshares = getchangeprofitshare(__projectdir__ / Path('temp/betas_adjustedmenu/'))

    plt.plot(betas, changeprofitshares)

    plt.xlabel('Beta')
    plt.ylabel('Change in Profit Share When Inflation Rises from 0 to 2\%')

    plt.tight_layout()

    plt.savefig(__projectdir__ / Path('temp/graphs/beta_adjustedmenu_changeprofitshare.png'))
    if show is True:
        plt.show()

    plt.clf()


def graph_sigma_adjustedmenu_changeprofitshare(show = False):
    menus, changeprofitshares = getchangeprofitshare(__projectdir__ / Path('temp/sigmas_adjustedmenu/'))

    plt.plot(menus, changeprofitshares)

    plt.xlabel('Sigma')
    plt.ylabel('Change in Profit Share When Inflation Rises from 0 to 2\%')

    plt.tight_layout()

    plt.savefig(__projectdir__ / Path('temp/graphs/sigma_adjustedmenu_changeprofitshare.png'))
    if show is True:
        plt.show()

    plt.clf()


def graph_menu_changeprofitshare(show = False):
    menus, changeprofitshares = getchangeprofitshare(__projectdir__ / Path('temp/menus/'))

    plt.plot(menus, changeprofitshares)

    plt.xlabel('Menu Cost')
    plt.ylabel('Change in Profit Share When Inflation Rises from 0 to 2\%)')

    plt.savefig(__projectdir__ / Path('temp/graphs/menu_changeprofitshare.png'))
    if show is True:
        plt.show()

    plt.clf()


def fullgraphs():
    if os.path.isdir(__projectdir__ / Path('temp/graphs/')):
        shutil.rmtree(__projectdir__ / Path('temp/graphs/'))
    os.mkdir(__projectdir__ / Path('temp/graphs/'))

    graph_pistar_profitshare()
    graph_pistar_pricechangeprob()
    graph_pistar_pricechangesize()
    graph_pistar_inactionsize()
    graph_beta_adjustedmenu_changeprofitshare()
    graph_sigma_adjustedmenu_changeprofitshare()
    graph_menu_changeprofitshare()


# Full:{{{1
def full():
    fullsavedicts()
    fullgraphs()
