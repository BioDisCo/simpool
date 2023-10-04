from Infections.Agent import Category
from scipy.stats import nbinom, poisson, lognorm
import math

# ------------------------- overall parameters -------------------------------------

# plot graphics at the end
plotting = True
# output single run statistics
output_singleruns = True
# output summary statistics
output_summary = True

# times to run simulation
Nsim = 10
# days simulation
T = 31*3
# T = 15

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
#N = 20 + 40  # 20 meds, 40 nursing staff
# beds = 176 # (at moment half used)

nb_agents = {
    Category.PATIENT:29,
    Category.NURSE: 27,
    Category.DOCTOR: 11,
    Category.ADMIN: 8,
}

# --------- smaller tests ----------

# --- enable for N = 2

"""
nb_agents = {
    Category.PATIENT: 0,
    Category.NURSE: 0,
    Category.DOCTOR: 2,
    Category.ADMIN: 0,
}
"""

# --- enable for N = 4

"""
nb_agents = {
    Category.PATIENT: 0,
    Category.NURSE: 2,
    Category.DOCTOR: 2,
    Category.ADMIN: 0,
}
"""

# --- enable for N = 10


nb_agents = {
   Category.PATIENT: 5,
   Category.NURSE: 2,
   Category.DOCTOR: 2,
   Category.ADMIN: 1,
}



# --------- end tests ------------



N = sum(nb_agents.values())

# P that someone is infected from outside (includes infection by patients)
# per day
pext = 0.01 * 0.05  # ?

# P that someone is infected by an infected contact
# per contact and day
pint = 0.002  # ?

# source: How best to use limited tests? Improving COVID-19 surveillance in long-term care (doi:10.1101/2020.04.19.20071639)
pinf_rate_permin = 0.0014
pinf_max_duration = 60
# P of internal infection, given the contact duration
def pinf(duration_min):
    return 1 - math.exp(-pinf_rate_permin*min(duration_min, pinf_max_duration))


# durration in min from where on contacts
# are considered as such (e.g. in contact tracing)
DURATION_TH = 15

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
usetest = mytests[3]

# -------------------------------------------------------------------------------

# number staff (dont change)
person_ids = list(range(N))

# kernel distributions
kerneldist = {
    Category.DOCTOR: {
        Category.DOCTOR: lambda: nbinom.rvs(**{'n': 0.639455389646031, 'p': 0.33441115478175726}),
        Category.ADMIN: lambda: nbinom.rvs(**{'n': 10.000001921571586, 'p': 0.990990726325269}),
        Category.NURSE: lambda: nbinom.rvs(**{'n': 10.0, 'p': 1.0}),
        Category.PATIENT: lambda: nbinom.rvs(**{'n': 10.0, 'p': 1.0}),
    },
    Category.NURSE: {
        Category.DOCTOR: lambda: nbinom.rvs(**{'n': 10.000004070316532, 'p': 1.0}),
        Category.ADMIN: lambda: nbinom.rvs(**{'n': 0.15085122247570856, 'p': 0.44891340197014196}),
        Category.NURSE: lambda: nbinom.rvs(**{'n': 0.4451399065297341, 'p': 0.30032845382338796}),
        Category.PATIENT: lambda: nbinom.rvs(**{'n': 0.11391011540643767, 'p': 0.2352151512186682}),
    },
}

contactsdist = {
    Category.DOCTOR: {
        Category.DOCTOR: lambda: nbinom.rvs(**{'n': 10.979419508176825, 'p': 0.7434157920988455}),
        Category.ADMIN: lambda: nbinom.rvs(**{'n': 198.4879963236352, 'p': 0.9939559641105772}),
        Category.NURSE: lambda: poisson.rvs(**{'mu': 2.002999988953658}),
        Category.PATIENT: lambda: nbinom.rvs(**{'n': 134.7317660804801, 'p': 0.987906385925085}),
    },
    Category.NURSE: {
        Category.DOCTOR: lambda: nbinom.rvs(**{'n': 143.0480506968438, 'p': 0.9896228228165873}),
        Category.ADMIN: lambda: nbinom.rvs(**{'n': 174.4605547357785, 'p': 0.9920733339719905}),
        Category.NURSE: lambda: nbinom.rvs(**{'n': 3.551790950529168, 'p': 0.45740247406029183}),
        Category.PATIENT: lambda: nbinom.rvs(**{'n': 5.456145786428396, 'p': 0.6579607647746138}),
    }
}

contact_length_dist = lambda: lognorm.rvs(*(1.4171168017880222, 4.812060883472516, 4.940292742938027))