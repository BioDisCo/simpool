#!/usr/bin/env python3

from enum import Enum, auto
import random
import numpy as np
from scipy import stats
import pylab as pl
import math


class State(Enum):
    """
    Agent infection states.

    UNINFECTED
    INFECTED_NONSPREADING
    INFECTED_SPREADING
    IMMUNE
    DEAD 
    """
    UNINFECTED = auto()
    INFECTED_NONSPREADING = auto()
    INFECTED_SPREADING = auto()
    IMMUNE = auto()
    DEAD = auto()

    def __str__(self):
        if self is State.UNINFECTED:
            return 'u'
        elif self is State.INFECTED_NONSPREADING:
            return 'i'
        elif self is State.INFECTED_SPREADING:
            return 's'
        elif self is State.IMMUNE:
            return 'm'
        elif self is State.DEAD:
            return 'd'
        else:
            raise ValueError('Unknown State')


class Symptoms(Enum):
    """
    Symptom severity an infected agent may have.

    ASYMPTOMATIC
    MILD
    SEVERE
    """
    ASYMPTOMATIC = auto()
    MILD = auto()
    SEVERE = auto()

    def __str__(self):
        if self is Symptoms.ASYMPTOMATIC:
            return ''
        elif self is Symptoms.MILD:
            return '~'
        elif self is Symptoms.SEVERE:
            return '!'
        else:
            raise ValueError('Unknown Symptom')


# P( death | severe symptomatic )
#
# source:
#   Fundamental principles of epidemic spread ... SARS-CoV-2 epidemic (Lourenco et al.)
#   Table 1: \theta
# model:
#  mean of \theta
pdeath = 0.14  # ?


# P( severe symptomatic | infected )
# P(        symptomatic | infected ) = 0.5 from island study (update!)
# P( severe symptomatic | symptomatic ) = 1 - 0.89 from Austrian news paper (update!)
#
# also discussed in Fundamental principles of epidemic spread ... SARS-CoV-2 epidemic (Lourenco et al.)
#   Table 1: \rho
# but cannot be fixed within a very wide range
# psyptomatic = 0.5 * (1 - 0.89)
# -- for the moment try this:
psyptomatic = 0.5 * (1 - 0.89)


# ---- latent period -----------
# = days from infection to spreading (i->s)
# distribution:
#  source:
#  https://www.medrxiv.org/content/10.1101/2020.03.21.20040329v1.full.pdf
#  p20 (right top)
# model:
#  seems lognormal, but the pdf does not match lognormal(2.52, 3.95)
#  used discrete but slightly adapted to better match fitted lognormal
pmin = 1/11
latentperiod_pk = (2*pmin, 4*pmin, 2*pmin, 1*pmin, 1*pmin, 1*pmin)
latentperiod_xk = np.arange(len(latentperiod_pk))
latentperiod_pdf = stats.rv_discrete(
    name='custm', values=(latentperiod_xk, latentperiod_pk))


def i2s_days():
    return math.floor(latentperiod_pdf.rvs(size=1)[0])


# ---- incubation period -----------
# = days from infection to symptoms, given that symptoms occur
# distribution:
#  source:
#  The incubation period of Coronavirus Disease 2019 (COVID-19) ... (Lauer)
#  https://annals.org/AIM/FULLARTICLE/2762808/INCUBATION-PERIOD-CORONAVIRUS-DISEASE-2019-COVID-19-FROM-PUBLICLY-REPORTED
#  -- attention! --
#  Figure 2 shape of cdf and the lognorm parameters did not fit for me.
#  And are incosinstent with Fig 2 caption and Appendix Table 2.
#  So I chose Appendix Table 2, Gamma distribution fit. This looks good comparing
#  to Figure 2.
# model:
#  Gamma()
shape = 5.807
scale = 0.948
latentperiod_dist = stats.gamma(a=shape, scale=scale)


def i2symp_days():
    return math.floor(latentperiod_dist.rvs(size=1)[0])

# ---- onset of viral spreading to virus negative -----------
# = length of period where infectious [days]
#
# distribution:
#  distribution different for mild and severe cases
#  -> we distinguish this, too
#  source:
#   Viral dynamics in mild and severe cases of COVID-19
#   https://www.thelancet.com/action/showPdf?pii=S1473-3099%2820%2930232-2
#  observaation:
#  days from onset of symptoms (?? check) (not infection) to virus negative
#  - severe case: positive after 10 days (from Figure: ~20 days )
#  - mild case: negative after <= 10 days (from Figure: ~10 days )
# model:
#  ! needs to be refined later on
#  at the moment constant times


def s2m_days(severe=True):
    """
    Returns number of days from symptomatic until ...

    Arguments:
    severe -- if severely ill (bool)
    """
    if severe:
        d = 20
    else:
        d = 10
    return d


class Agent:

    def __init__(self, myid, myinfex):
        """
        Creates an agent.

        Arguments:
        myid -- id of the agent (int)
        myinfex -- infection execution for logging (InfectionExecution)
        """
        self.id = myid

        self.state = State.UNINFECTED
        self.state_timer = -float('inf')
        self.next_state = State.UNINFECTED

        self.quarantined = False
        self.quarantine_timer = -float('inf')

        self.time_infected = -float('inf')
        self.infected_by = None

        self.symptoms = None
        self.symptoms_timer = -float('inf')
        self.severe_symptomatic = False
        self.severe_symptomatic_timer = -float('inf')

        self.myinfex = myinfex

    def __str__(self):
        mystr = str(self.symptoms)
        # FIXME: remove next line
        mystr += '!' if self.severe_symptomatic else ''
        mystr += 'q' if self.quarantined else ''
        mystr += str(self.state)
        return mystr

    def infect(self, by_id, time):
        """
        Infect agent by agent with id <by_id> at time <time>

        Arguments:
        by_id -- id of agent that infects this agent (int)
        time -- time of infection (int)
        """
        if self.state is State.UNINFECTED:
            self.state = State.INFECTED_NONSPREADING
            self.state_timer = i2s_days()
            self.next_state = State.INFECTED_SPREADING
            self.time_infected = time
            self.infected_by = by_id

            # choose if will be severe symptomatic and if so when
            u = random.random()
            if u <= psyptomatic:
                self.severe_symptomatic_timer = i2symp_days()

    def quarantine(self, days):
        """
        Quarantines agent for days from now.

        Arguments:
        days -- number of days to quarantine from now (int)
        """
        self.quarantined = True
        self.quarantine_timer = days
        # print(f'quarantine {self.id}')

    def dequarantine(self):
        """
        Dequarantines agent.
        """
        self.quarantined = False
        self.quarantine_timer = -float('inf')
        # print(f'dequarantine {self.id}')

    def works(self):
        """
        Returns if agent works.
        """
        # if works in hospital
        if self.state is State.DEAD or self.quarantined:
            return False
        elif self.severe_symptomatic:
            return False
        else:
            return True

    def tick(self, t):
        """
        Performs a one day state change of the agent.

        Arguments:
        t -- current time (int)
        """
        self.state_timer -= 1

        if self.state is State.INFECTED_NONSPREADING:
            if self.state_timer == 0:
                self.state = self.next_state
                if self.severe_symptomatic_timer > -float('inf'):
                    # this agent will be severe symptomatic
                    # time until (severe) symptoms
                    self.severe_symptomatic_timer = i2symp_days()
                    # print(self.severe_symptomatic_timer)
                    # check if will die
                    u = random.random()
                    if u <= pdeath:
                        # will die
                        self.next_state = State.DEAD
                        # ! check this later -> here assumed that death distributed
                        #   like virus negative
                        self.state_timer = s2m_days(severe=True)
                    else:
                        # will not die, but severe symptoms
                        self.next_state = State.IMMUNE
                        self.state_timer = s2m_days(severe=True)
                else:
                    # will not die, and only mild symptoms (if any)
                    self.next_state = State.IMMUNE
                    self.state_timer = s2m_days(severe=False)
        elif self.state is State.INFECTED_SPREADING:
            if self.state_timer == 0:
                self.state = self.next_state

        self.quarantine_timer -= 1
        if self.quarantined and self.quarantine_timer == 0:
            self.dequarantine()

        if self.severe_symptomatic:
            # TODO: implement me better
            # assumption currently:
            # State.IMMUNE -> self.severe_symptomatic = False
            if self.state is State.IMMUNE:
                self.severe_symptomatic = False
                self.myinfex.remove_symptomatic(t, self.id)
        else:
            # not yet severe symptoms
            self.severe_symptomatic_timer -= 1
            if self.severe_symptomatic_timer == 0:
                self.severe_symptomatic = True
                # print(f'{self.id} got sympt')
                self.myinfex.add_symptomatic(t, self.id)
