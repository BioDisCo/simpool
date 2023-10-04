#!/usr/bin/env python3

import matplotlib.pyplot as plt

import simple
import config
import numpy as np

# tests to compare:
mytests = {}
mytests[0] = {
    'short': 'No test',
    'type': 'TestJustLook',
    'parameters': {}
}
mytests[1] = {
    'short': 'All individually',
    'type': 'TestAll',
    'parameters': {}
}
mytests[2] = {
    'short': 'Contact trace',
    'type': 'TestContacts',
    'parameters': {
        'individual_quarantine_days': 2*7,
        # contact of positives are quarantined for 14 days AND no symptoms
        'contact_quarantine_days': 2*7,
        'lookback_days': 2,                 # 2 days before symptom onset
        #
        'contacts_check': False,            # don't check these right away
        'contacts_poolsize': 5,             #
            'contacts_k_group': 1,              #
            #
        'working_check': False,             # don't check all others-> ignore next
        'working_quarantine_days': 2*7,
        'working_poolsize': 5,              #
            'working_k_group': 1,               #
            #
        # if positive in quarantine, retest in 7 days (check!)
        'days_if_positive': 7,
        'k_release': 2,                     # 2 negative tests required before release
        'days_after_negative': 2}           # after tests are negative, wait another 2 days (check!)
}
mytests[3] = {
    'short': 'Pooled (10)',
    'type': 'TestPool',
    'parameters': {
            'individual_quarantine_days': 2*7,  # quarantine for days if individual pos
            #
            'group_quarantine_days': 0,        # quarantine group for days if group pos
            'poolsize': 10,                     # uniform poolsize
            'k_group': 1,                      # k for group test
            #
            'days_if_positive': 7,             # if positive again, schedule next test in days
            'k_release': 1,                    # k for release test
        'days_after_negative': 0}          # days after negative test still in quarantine
}


# main
if __name__ == "__main__":
    # turn off standard output
    config.plotting = False
    config.output_singleruns = False
    config.output_summary = False

    # choose tests to compare
    to_compare = [
        mytests[0],
        mytests[1],
        mytests[2],
        mytests[3]]

    # generate data for each test
    perfs = []
    for tidx in range(len(to_compare)):
        usetest = to_compare[tidx]
        print(f"#{usetest}")
        print(f"{tidx}")
        perfs += [ simple.get_performance(usetest=usetest) ]

    # comparison plots
    strategy_names = [s['short'] for s in to_compare]

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel('test strategy')
    ax.set_ylabel('# of persons')
    ax.set_title('Deceased persons')
    data = [ p['seq_death_nb'] for p in perfs ]
    avgs = [np.average(d) for d in data]
    stds = [np.std(d) for d in data]
    x = list(range(len(avgs)))
    ax.bar(x, avgs, yerr=stds, capsize=15)
    ax.set_ybound(lower=0, upper=5)
    plt.xticks(x, strategy_names)
    plt.savefig('fig_Deceased_persons.pdf')

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel('test strategy')
    ax.set_ylabel('# tests per day')
    ax.set_title('Average number of tests per day')
    data = [ p['seq_testsperday'] for p in perfs ]
    avgs = [np.average(d) for d in data]
    stds = [np.std(d) for d in data]
    x = list(range(len(avgs)))
    ax.bar(x, avgs, yerr=stds, capsize=15)
    ax.set_ybound(lower=0, upper=config.N)
    plt.xticks(x, strategy_names)
    plt.savefig('fig_Average_number_of_tests_per_day.pdf')

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel('test strategy')
    ax.set_ylabel('# tests')
    ax.set_title('Maximum number of tests on a day')
    data = [ p['seq_max_testsonday'] for p in perfs ]
    avgs = [np.average(d) for d in data]
    stds = [np.std(d) for d in data]
    x = list(range(len(avgs)))
    ax.bar(x, avgs, yerr=stds, capsize=15)
    ax.set_ybound(lower=0, upper=config.N)
    plt.xticks(x, strategy_names)
    plt.savefig('fig_Maximum_number_of_tests_per_day.pdf')

    plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel('test strategy')
    ax.set_ylabel('# spreading days')
    ax.set_title('Cumulative number of working spreading personnel')
    data = [ p['seq_spreading_days'] for p in perfs ]
    avgs = [np.average(d) for d in data]
    stds = [np.std(d) for d in data]
    x = list(range(len(avgs)))
    ax.bar(x, avgs, yerr=stds, capsize=15)
    ax.set_ybound(lower=0, upper=config.N*15)
    plt.xticks(x, strategy_names)
    plt.savefig('fig_Cumulative_number_of_working_spreading_personnel.pdf')

    # working per day (averaged over all simulations)
    plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel('day')
    ax.set_ylabel('# of persons')
    ax.set_title('Daywise average working personnel')
    data = [ p['seq_seq_avg_working'] for p in perfs ]
    T = len(data[0])
    x = range(T)
    colors = 'rmbg'
    for i, d in enumerate(data):
        ax.plot(x, d, '-' + colors[i % len(colors)], label=strategy_names[i])
    ax.set_ybound(lower=0, upper=config.N)
    ax.legend(shadow=False, ncol=1, frameon=False)
    plt.xticks(range(0,T,5), range(0,T,5))
    plt.savefig('fig_Daywise_average_working_personnel.pdf')

    plt.show()
