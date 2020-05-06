# ------------------------- overall parameters -------------------------------------

# plot graphics
plotting = True
# output singe run statistics
output_singleruns = True
# output summary statistics
output_summary = True

# times to run simulation
Nsim = 100
# days simulation
T = 31*3

# each person names the persons it interact with the most
# at the moment say all
# op 7
# meeting 5
# station some
# nursing staff 6
# meds 10
# overall contact estimate >= 30 but < 40
# contacts_perday = 30
# -- look only at close contacts at the moment
# close risk contacts / day and person [pers comm, Barbara]
contacts_perday = 4

# -- hospital --
# N = 1300
# beds = 1050 # (at moment half used)

# -- station --
N = 20 + 40  # 20 meds, 40 nursing staff
# beds = 176 # (at moment half used)

# P that someone is infected from outside (includes infection by patients)
# per day
pext = 0.01 * 0.5  # ?

# P that someone is infected by an infected contact
# per contact and day
pint = 0.05  # ?

# choose test:
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

# chose among above or new ones
usetest = mytests[1]

# -------------------------------------------------------------------------------

# number staff (dont change)
person_ids = list(range(N))
