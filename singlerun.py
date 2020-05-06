#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import simple
import config
from Infections.ExecutionPlotter import ExecutionPlotter


# tests:
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
    'short': 'Pooled (5)',
    'type': 'TestPool',
    'parameters': {
            'individual_quarantine_days': 2*7,  # quarantine for days if individual pos
            #
            'group_quarantine_days': 2,        # quarantine group for days if group pos
            'poolsize': 5,                     # uniform poolsize
            'k_group': 1,                      # k for group test
            #
            'days_if_positive': 7,             # if positive again, schedule next test in days
            'k_release': 2,                    # k for release test
        'days_after_negative': 2}          # days after negative test still in quarantine
}


# main
if __name__ == "__main__":
    # turn off standard output
    config.plotting = False
    config.output_singleruns = False
    config.output_summary = False

    usetest = mytests[2]
    u, i, s, m, q, d, working, infex, test, state_vec, infex = simple.run_sim(usetest=usetest)

    ep = ExecutionPlotter(config.N, config.T)
    for i in config.person_ids:
        ep.add_agent(infex, agent_id=i)
    ep.sort()    
    ep.plot(title=usetest['short'], save=True)

    plt.show()
