import simple
import config

# tests to compare:
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

    # choose tests to compare
    to_compare = [
        mytests[0],
        mytests[1],
        mytests[2],
        mytests[3]]

    # how many samples per test
    datapoints_per_comparison = 20

    # generate data for each test
    for tidx in range(len(to_compare)):
        usetest = to_compare[tidx]
        print(f"#{usetest}")
        print(f"{tidx}")
        for p in range(datapoints_per_comparison):
            perf = simple.get_performance(usetest=usetest)
            print(f'{tidx},{perf[0]},{perf[1]},{perf[2]},{perf[3]}')
        print("")
