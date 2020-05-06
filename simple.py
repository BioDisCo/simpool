#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import config

from Infections.InfectionExecution import InfectionExecution
from Infections.Agent import Agent
from Infections.Agent import State
from Infections.Test import get_Test
from Infections.Test import PoolableProbing
from Infections.Test import TestAll, TestJustLook, TestPool, TestContacts
from Infections.Hospital import Hospital


def run_sim(usetest, return_state_vec=False):
    """
    run a single simulation
    """
    # sim
    infex = InfectionExecution(N=config.N)
    state = [Agent(i, infex) for i in range(config.N)]
    u = []
    i = []
    s = []
    m = []
    q = []
    d = []
    working = []
    if return_state_vec:
        state_vec = []
    else:
        state_vec = None

    # set probing and test
    probing = PoolableProbing()
    test = get_Test(probing=probing, infex=infex, usetest=usetest)

    for t in range(config.T):
        # print(f'time: {t}')
        # print(f'{t}: {state}')

        # log per class
        classes_working = [s.state for s in state if s.works()]
        classes_all = [s.state for s in state]
        quarantined = [s.quarantined for s in state]
        u += [classes_working.count(State.UNINFECTED)]
        i += [classes_working.count(State.INFECTED_NONSPREADING)]
        s += [classes_working.count(State.INFECTED_SPREADING)]
        m += [classes_working.count(State.IMMUNE)]
        q += [quarantined.count(True)]
        d += [classes_all.count(State.DEAD)]
        is_working = [s.works() for s in state]
        working += [is_working.count(True)]

        # conditinally return state vector
        if return_state_vec:
            state_vec += [classes_all]

        # -- to print traces --
        # print(list(map(str, state)))

        # transition
        for person in config.person_ids:
            infex.next_state(t, person, state)

        # test
        test.do_test(state, t)

    return u, i, s, m, q, d, working, infex, test, state_vec


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

    for k in range(config.Nsim):
        u, i, s, m, q, d, working, infex, test, state_vec = run_sim(usetest)
        if config.plotting:
            if k > 0:
                alpha = 0.02
                ax.plot(u, 'g-', alpha=alpha)
                ax.plot(i, 'm-', alpha=alpha)
                ax.plot(s, 'r-', alpha=alpha)
                ax.plot(m, 'b-', alpha=alpha)
                ax.plot(q, 'y-', alpha=alpha)
                ax.plot(d, 'k-', alpha=alpha)
                ax2.plot(working, 'g-', alpha=alpha)
                ax2.plot(d, 'k-', alpha=alpha)
            else:
                # bold first plot
                alpha = 1.0
                ax.plot(u, 'g-', label='working uninfected', alpha=alpha)
                ax.plot(i, 'm-', label='working infected non-spreading', alpha=alpha)
                ax.plot(s, 'r-', label='working infected spreading', alpha=alpha)
                ax.plot(m, 'b-', label='working recovered immune', alpha=alpha)
                ax.plot(q, 'y-', label='quarantined', alpha=alpha)
                ax.plot(d, 'k-', label='deceased', alpha=alpha)
                ax2.plot(working, 'g-', label='working total', alpha=alpha)
                ax2.plot(d, 'k-', label='deceased', alpha=alpha)
            infex.plot(ax3, alpha=alpha)
            infex.plotparam(axp2, 'R', alpha=alpha)
            infex.plotparam(axp3, 'serial', alpha=alpha)

        # get performance
        died_frac = d[-1] / config.N
        minwork_frac = min(working) / config.N
        uninfected_frac = u[-1] / config.N
        tests_eachday = test.get_nb_tests()
        # log
        overall_death_frac += [died_frac]
        overall_minwork_frac += [minwork_frac]
        overall_uninfected_frac += [uninfected_frac]
        overall_tests_perday_avg += [np.average(tests_eachday)]
        overall_tests_perday_max += [max(tests_eachday)]
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
        ax.set_ybound(lower=0, upper=config.N)

        fig1.subplots_adjust(top=0.85)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), ncol=2,
                  borderaxespad=0, frameon=False)

        ax2.set_xbound(lower=0, upper=config.T)
        ax2.set_ybound(lower=0, upper=config.N)

        ax3.set_xbound(lower=0, upper=config.T)
        ax3.set_ybound(lower=0, upper=config.N)

        axp1.set_xbound(lower=0, upper=0.05*100)
        axp1.set_ybound(lower=0, upper=1*100)
        axp1.set_xlabel('deceased [%]')
        axp1.set_ylabel('minimum working staff/day [%]')

        # switch on for legends
        # ax.legend(shadow=True, ncol=2)
        ax2.legend(shadow=False, ncol=1, frameon=False)

        plt.show()

    # return tuple (avg died[pers], avg minwork[pers], avg tests/day, avg maximal tests/day )
    return (np.average(overall_death_frac)*config.N,
            np.average(overall_minwork_frac)*config.N,
            np.average(overall_tests_perday_avg),
            np.average(overall_tests_perday_max))


# main
if __name__ == "__main__":
    perf = get_performance(usetest=config.usetest)
    print(perf)
