#!/usr/bin/env python3

from enum import Enum, auto
import random
import numpy as np
from scipy import stats
import pylab as pl
import math
from scipy.special import comb

class EventQueue:
    def __init__(self):
        self.queue = []

    def enqueue(self, delay: int, event):
        assert(delay >= 0)
        self.queue += [(delay, event)]

    def dequeue(self):
        """
        Decrements all timers and returns next day's event list.
        """
        today = filter(lambda x: x[0] <= 1, self.queue)
        self.queue = filter(lambda x: x[0] > 1, self.queue)
        self.queue = list(map(lambda x: [x[0]-1, x[1]], self.queue))
        return [ x[1] for x in today ]

    def clear(self):
        self.queue = []

class Symptoms(Enum):
    """
    Agent's severity of symptoms.

    NONE
    MILD
    SEVERE
    DEAD 
    """
    NONE = auto()
    MILD = auto()
    SEVERE = auto()
    DEAD = auto()

    def __str__(self) -> str:
        if self is Symptoms.NONE:
            return 'n'
        elif self is Symptoms.MILD:
            return 'm'
        elif self is Symptoms.SEVERE:
            return 's'
        elif self is Symptoms.DEAD:
            return 'd'
        else:
            raise ValueError('Unknown State')



# P( death | severe symptomatic )
#
# source:
#   Fundamental principles of epidemic spread ... SARS-CoV-2 epidemic (Lourenco et al.)
#   Table 1: \theta
# model:
#  mean of \theta
p_death = {Symptoms.MILD: 0.0, Symptoms.SEVERE: 0.14}

# P( severe symptomatic | infected )
# P(        symptomatic | infected ) = 0.5 from island study (update!)
# P( severe symptomatic | symptomatic ) = 1 - 0.89 from Austrian news paper (update!)
#
# also discussed in Fundamental principles of epidemic spread ... SARS-CoV-2 epidemic (Lourenco et al.)
#   Table 1: \rho
# but cannot be fixed within a very wide range
# psyptomatic = 0.5 * (1 - 0.89)
# -- for the moment try this:
p_symptomatic = 0.5
p_severe_syptomatic = p_symptomatic * (1 - 0.89)
p_mild_symptomatic = p_symptomatic - p_severe_syptomatic


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

def symptoms_delay(symptoms):
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
#  - severe case: positive after 20 days (from Figure: ~20 days )
#  - mild case: negative after <= 10 days (from Figure: ~10 days )
# model:
#  ! needs to be refined later on
#  at the moment constant times


def severity_days(symptoms):
    """
    Returns number of days for an agent
    with symptoms <symptoms>

    Arguments:
    symptoms -- agent's symptoms (Symptoms)
    """
    if symptoms is Symptoms.MILD:
        d = 20
    elif symptoms is Symptoms.SEVERE:
        d = 20
    elif symptoms is Symptoms.NONE:
        d = 20 #FIXME: guess
    else:
        raise ValueError("invalid symptoms")
    return d


# RNA levels.
# based on rough shape in https://www.nature.com/articles/s41591-020-0869-5#Fig3
# but needs to be adjusted to the data.
# deterministic at the moment.
def relative_rna_levels(days_until_symptomonset: int, symptoms: Symptoms) -> list:
    #TODO: refine model
    p = 0.1
    # -2 days before potential onset: no RNA
    rna_levels = [0 for d in range(1,days_until_symptomonset-2)]
    # then:
    for d in range(20):
        r = comb(40,d) *  p**d * (1-p)**d / (comb(40,3) *  p**3 * (1-p)**3)
        rna_levels += [r]
    # finally:
    rna_levels += [0]
    return rna_levels



def draw_course_of_disease():
    # draw severity
    u = random.random()
    if u >= p_symptomatic:
        symptoms = Symptoms.NONE
    elif u >= p_mild_symptomatic:
        symptoms = Symptoms.SEVERE
    else:
        symptoms = Symptoms.MILD

    symptoms_onset = symptoms_delay(symptoms)
    symptoms_duration = severity_days(symptoms)

    symptoms_events = []

    # incubation
    if symptoms is not Symptoms.NONE:
        
        symptoms_events += [(symptoms_onset, symptoms)]

        # dies?
        u = random.random()
        if u < p_death[symptoms]:
            symptoms_events += [(symptoms_onset+symptoms_duration, Symptoms.DEAD)]
        else:
            symptoms_events += [(symptoms_onset+symptoms_duration, Symptoms.NONE)]

    # RNA levels
    rna_levels = relative_rna_levels(days_until_symptomonset= symptoms_onset, symptoms= symptoms)
    rna_events = [(d+1,r) for d,r in enumerate(rna_levels)]

    return symptoms_events, rna_events




class Agent:

    def __init__(self, myid, myinfex):
        """
        Creates an agent.

        Arguments:
        myid -- id of the agent (int)
        myinfex -- infection execution for logging (InfectionExecution)
        """
        self.id = myid

        self.infected = False
        self.infected_by = None

        self.symptoms = Symptoms.NONE
        self.symptoms_queue = EventQueue()

        self.rna = 0
        self.rna_queue = EventQueue()

        self.quarantined = False
        self.quarantine_queue = EventQueue()

        self.myinfex = myinfex

    @property
    def spreading(self):
        return self.rna > 0

    def __str__(self) -> str:
        mystr = str(self.symptoms)
        # FIXME: remove next line
        mystr += 'q' if self.quarantined else ''
        return mystr

    @property
    def state(self):
        return {'infected': self.infected,
                'infected_by': self.infected_by,
                'symptoms': self.symptoms,
                'rna': self.rna,
                'quarantined': self.quarantined,
                'working': self.working,
                'dead': self.dead,
                'sick_leave': self.sick_leave,
                'spreading': self.spreading,
                }

    def infect(self, by_id, time):
        """
        Infect agent by agent with id <by_id> at time <time>

        Arguments:
        by_id -- id of agent that infects this agent (int)
        time -- time of infection (int)
        """
        if not self.infected:
            symptoms_events, rna_events = draw_course_of_disease()

            for d,e in symptoms_events:
                self.symptoms_queue.enqueue(d,e)
            for d,e in rna_events:
                self.rna_queue.enqueue(d,e)

            self.infected = True
            self.infected_by = by_id
            self.time_infected = time

    def quarantine(self, days: int):
        """
        Quarantines agent for days from now.

        Arguments:
        days -- number of days to quarantine from now (int)
        """
        assert(days >= 0)
        if days > 0:
            self.quarantined = True
            self.quarantine_queue.clear()
            self.quarantine_queue.enqueue(days, False)
        else:
            # do not quarantine
            pass

    def dequarantine(self):
        """
        Dequarantines agent.
        """
        self.quarantined = False
        self.quarantine_queue.clear()

    @property
    def working(self):
        """
        If agent works.
        """
        return self.symptoms is Symptoms.NONE and not self.quarantined 

    @property
    def dead(self):
        """
        If agent is dead.
        """
        return self.symptoms is Symptoms.DEAD

    @property
    def sick_leave(self):
        """
        If agent is alive, but too sick to work.
        """
        return self.symptoms in {Symptoms.MILD, Symptoms.SEVERE}

    def tick(self):
        """
        Performs a one day state change of the agent.
        """

        symptoms_events = self.symptoms_queue.dequeue()
        assert(len(symptoms_events) <= 1)
        for e in symptoms_events:
            self.symptoms = e

        rna_events = self.rna_queue.dequeue()
        assert(len(rna_events) <= 1)
        for e in rna_events:
            self.rna = e

        quarantine_events = self.quarantine_queue.dequeue()
        assert(len(quarantine_events) <= 1)
        for e in quarantine_events:
            self.quarantined = e