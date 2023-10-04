import simple
import config

# tests:
mytests = {}
mytests[0] = {
    'type': 'TestAll',
    'parameters': {}
}
mytests[1] = {
    'type': 'TestJustLook',
    'parameters': {}
}
mytests[2] = {
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
mytests[3] = {
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


# main
if __name__ == "__main__":
    # turn off standard output
    config.plotting = False
    config.output_singleruns = False
    config.output_summary = False

    # choose test
    usetest = mytests[2]
    # parameter
    sweep_parameter = 'poolsize'
    sweep_range = range(0, 10, 1)

    # sweep the test
    print(f"#{usetest}")
    print(
        f"#{sweep_parameter},avg_died[pers],avg_minwork[pers],avg_tests/day,avg_maximal_tests/day")
    for p in sweep_range:
        usetest[sweep_parameter] = p
        perf = simple.get_performance(usetest=usetest)
        print(f'{p},{perf[0]},{perf[1]},{perf[2]},{perf[3]}')
