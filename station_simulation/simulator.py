#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import config

from Infections.InfectionExecution import InfectionExecution
from Infections.Agent import Agent
from Infections.Agent import Symptoms
from Infections.Test import get_Test
from Infections.Test import PoolableProbing
from Infections.Test import TestAll, TestJustLook, TestPool, TestContacts
from Infections.Hospital import Hospital



def run_sim(usetest, return_state_vec=False, initial_infections={'T_range': (0,10), 'N_range': (0,2)}):
    """
    run a single simulation
    """
    # sim
    infex = InfectionExecution(initial_infections=initial_infections)
    results = {
        'uninfected': [],
        'infected': [],
        'spreading': [],
        'immune': [],
        'quarantined': [],
        'deceased': [],
        'working': [],
        'staff': [],
    }
    state_vec = None
    # set probing and test
    probing = PoolableProbing()
    test = get_Test(probing=probing, infex=infex, usetest=usetest)
    # iterate days
    for t in range(config.T):
        # -- end of previous day
        infex.tick_endofworkingday()
        # also log classes
        results['uninfected'] += [infex.get_nb_staff(t) - infex.get_nb_infected(t)]
        results['infected'] += [infex.get_nb_working(t) - infex.get_nb_working_spreading(t)]
        results['spreading'] += [infex.get_nb_working_spreading(t)]
        results['immune'] += [infex.get_nb_working_immune(t)]
        results['quarantined'] += [infex.get_nb_quarantined(t)]
        results['deceased'] += [infex.get_nb_dead(t)]
        results['working'] += [infex.get_nb_working(t)]
        results['staff'] += [infex.get_nb_staff(t)]
        
        # -- next day
        # next day and get tested
        infex.tick_nextday()
        test.do_test()
        # start working
        infex.tick_startworkingday()

    # conditinally return state vector
    if return_state_vec:
        state_vec = infex.states
    return results, infex, test, state_vec


def get_performance(usetest):
    """
    run several simulations with the specified test.

    Arguments:
    Test as 'usetest'

    Returns:
    Performance as tuple
    """

    # plots
    if config.plotting:
        fig1 = plt.figure()
        ax = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        ax.set_ylabel('# of persons')
        ax2.set_ylabel('# of persons')
        ax3.set_xlabel('time [days]')
        ax3.set_ylabel('person ID')

        plt.figure()
        axp1 = plt.subplot(111)

        plt.figure()
        axp2 = plt.subplot(111)

        plt.figure()
        axp3 = plt.subplot(111)

    overall_death_frac = []
    overall_minwork_frac = []
    overall_uninfected_frac = []
    overall_tests_perday_avg = []
    overall_tests_perday_max = []
    overall_spreading_days_sum = []
    seq_seq_working = []

    for k in range(config.Nsim):
        results, infex, test, state_vec = run_sim(usetest)
        if config.plotting:
            if k > 0:
                alpha = 0.02
                ax.plot(results['uninfected'], 'g-', alpha=alpha)
                ax.plot(results['infected'], 'm-', alpha=alpha)
                ax.plot(results['spreading'], 'r-', alpha=alpha)
                ax.plot(results['immune'], 'b-', alpha=alpha)
                ax.plot(results['quarantined'], 'y-', alpha=alpha)
                ax.plot(results['deceased'], 'k-', alpha=alpha)
                
                ax2.plot(results['working'], 'g-', alpha=alpha)
                ax2.plot(results['deceased'], 'k-', alpha=alpha)
            else:
                # bold first plot
                alpha = 1.0
                ax.plot(results['uninfected'], 'g-', label='working uninfected', alpha=alpha)
                ax.plot(results['infected'], 'm-', label='working infected non-spreading', alpha=alpha)
                ax.plot(results['spreading'], 'r-', label='working infected spreading', alpha=alpha)
                ax.plot(results['immune'], 'b-', label='working recovered immune', alpha=alpha)
                ax.plot(results['quarantined'], 'y-', label='quarantined', alpha=alpha)
                ax.plot(results['deceased'], 'k-', label='deceased', alpha=alpha)

                ax2.plot(results['working'], 'g-', label='working total', alpha=alpha)
                ax2.plot(results['deceased'], 'k-', label='deceased', alpha=alpha)
            infex.plot(ax3, alpha=alpha)
            infex.plotparam(axp2, 'R', alpha=alpha)
            infex.plotparam(axp3, 'serial', alpha=alpha)

        # get performance
        N = results['staff'][-1]
        died_frac = results['deceased'][-1] / N
        minwork_frac = min(results['working']) / N
        uninfected_frac = results['uninfected'][-1] / N
        tests_eachday = test.get_nb_tests()
        # log
        overall_death_frac += [died_frac]
        overall_minwork_frac += [minwork_frac]
        overall_uninfected_frac += [uninfected_frac]
        overall_tests_perday_avg += [np.average(tests_eachday)]
        overall_tests_perday_max += [max(tests_eachday)]
        overall_spreading_days_sum += [sum(results['spreading'])]
        seq_seq_working += [results['working']] 
        # print
        if config.output_singleruns:
            print('')
            print(f'deceased[%]= {died_frac*100:12.2f}')
            print(f'minwork[%]= {minwork_frac*100:13.2f}')
            print(f'infected[%]= {(1-uninfected_frac)*100:12.2f}')
            print(f'avg tests/day= {overall_tests_perday_avg[-1]:10.2f}')
            print(f'maximal tests/day= {overall_tests_perday_max[-1]:6.2f}')
        # plot performance
        if config.plotting:
            axp1.plot([died_frac*100], [minwork_frac*100],
                      'r', marker='x', alpha=1.0)

    # overall print
    if config.output_summary:
        print('------------ test ------------------------------------')
        print(test)
        print('------------ summary ---------------------------------')
        print(
            f'avg deceased[%]= {np.average(overall_death_frac)*100:13.2f},  [pers] = {np.average(overall_death_frac)*config.N:6.2f}')
        print(
            f'avg minwork[%]= {np.average(overall_minwork_frac)*100:14.2f},  [pers] = {np.average(overall_minwork_frac)*config.N:6.2f}')
        print(
            f'avg infected[%]= {(1 - np.average(overall_uninfected_frac))*100:13.2f},  [pers] = {(1 - np.average(overall_uninfected_frac))*config.N:6.2f}')
        print(f'avg tests/day= {np.average(overall_tests_perday_avg):15.2f}')
        print(
            f'avg maximal tests/day= {np.average(overall_tests_perday_max):7.2f}')

    # overall plot
    if config.plotting:
        ax.set_xbound(lower=0, upper=config.T)
        ax.set_ybound(lower=0, upper=N)

        fig1.subplots_adjust(top=0.85)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=2,
                  borderaxespad=0, frameon=False)

        ax2.set_xbound(lower=0, upper=config.T)
        ax2.set_ybound(lower=0, upper=N)

        ax3.set_xbound(lower=0, upper=config.T)
        ax3.set_ybound(lower=0, upper=N)

        axp1.set_xbound(lower=0, upper=0.05*100)
        axp1.set_ybound(lower=0, upper=1*100)
        axp1.set_xlabel('deceased [%]')
        axp1.set_ylabel('minimum working staff/day [%]')

        # switch on for legends
        # ax.legend(shadow=True, ncol=2)
        ax2.legend(shadow=False, ncol=1, frameon=False)

        plt.show()

    # return performance
    return {
        'avg_death_nb': np.average(overall_death_frac)*config.N,
        'seq_death_nb': [x * config.N for x in overall_death_frac],
        'avg_minwork_nb': np.average(overall_minwork_frac)*config.N,
        'seq_minwork_nb': [x * config.N for x in overall_minwork_frac],
        'avg_testsperday': np.average(overall_tests_perday_avg),
        'seq_testsperday': overall_tests_perday_avg,
        'avg_max_testsonday': np.average(overall_tests_perday_max),
        'seq_max_testsonday': overall_tests_perday_max,
        'seq_spreading_days': overall_spreading_days_sum,
        'seq_seq_avg_working': np.average(seq_seq_working, axis=0), # point-wise average
    }


# main
if __name__ == "__main__":
    perf = get_performance(usetest=config.usetest)
    print(perf)
