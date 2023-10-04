#!/usr/bin/env python3
import sys
sys.path.append("../")

import config

import random
import numpy as np
from Infections.Agent import Agent
from Infections.Agent import Symptoms
from Infections.InfectionExecution import InfectionExecution
from itertools import zip_longest

from ML import learn_simple

test_perday = 1300

# append new tests here
REGISTERED_TESTS = ['TestAll',
    'TestJustLook', 
    'TestPool', 
    'TestContacts', 
    'TestRandom', 
    'TestNull', 
    'TestSpreadingPredictionNN']


def nb_available_tests(t):
    return test_perday


class PoolableProbing(object):
    """
    Probings are 'tests' that turn out pos or negative.
    These ones are poolable.

    Assumed here: detectable exactly when spreading.

    Examples: rt pcr, ...
    """

    #def __init__(self, se=0.65, sp=0.95, k=1):
    def __init__(self, se=1.0, sp=1.0, k=1):
        self.se = 1-(1-se)**k
        self.sp = 1-(1-sp)**k

    def __str__(self) -> str:
        return f"PoolableProbing(se={self.se:.3f}, sp={self.sp:.3f})"

    def __ispositive__(self, agent) -> bool:
        """
        Returns if <agent> is positive
        """
        return agent.rna > 0 #FIXME

    def probe(self, agents, k=1) -> bool:
        """
        Returns test result for pooled group <agents>
        """
        # determine real positivity of group <agents>
        group_positive = False
        for agent in agents:
            if self.__ispositive__(agent):
                group_positive = True
        # set local se and sp
        se = 1-(1-self.se)**k
        sp = 1-(1-self.sp)**k
        # test group
        u = random.random()
        if group_positive:
            if u <= se:
                return True
            else:
                return False
        else:
            if u > sp:
                return True
            else:
                return False


class Test(object):
    def __init__(self, probing: PoolableProbing, infex: InfectionExecution):
        self.nb_tests = []
        self.probing = probing
        self.infex = infex

    def __str__(self):
        return "Test"

    def __get_alive__(self):
        return list(filter(lambda i: not(self.infex.agents[i].symptoms is Symptoms.DEAD),
                           list(range(len(self.infex.agents)))))

    def get_nb_tests(self):
        return list(map(lambda tup: tup[1], self.nb_tests))

    def total_nb_tests(self):
        return sum(map(lambda tup: tup[1], self.nb_tests))

    def do_test(self):
        pass


class TestJustLook(Test):
    """
    Test strategy:
    - Do not test.
    - Only quarantine when someone gets sever symptoms.
      Then quarantine for 7 days.
    """

    def __init__(self, probing: PoolableProbing, infex: InfectionExecution):
        super().__init__(probing, infex)

    def __str__(self):
        return \
            f"""TestJustLook:
probing={self.probing},
infex={self.infex}"""

    def do_test(self):

        pop = config.person_ids
        for i in pop:
            if self.infex.agents[i].symptoms is Symptoms.SEVERE:
                # detect and quarantine each day by just looking
                # if it is severe
                self.infex.agents[i].quarantine(days=7)

        self.nb_tests += [(self.infex.t, 0)]


class TestAll(Test):
    """
    Test strategy:
    - Test each person, each day.
    - If positive, send to quarantine for 3 weeks.
    """

    def __init__(self, probing: PoolableProbing, infex: InfectionExecution):
        super().__init__(probing, infex)

    def __str__(self):
        return \
            f"""TestAll:
probing={self.probing},
infex={self.infex}"""

    def do_test(self):
        tests_max = nb_available_tests(self.infex.t)
        cnt = 0

        pop = config.person_ids
        random.shuffle(pop)
        for i in pop:
            if self.infex.agents[i].symptoms is Symptoms.DEAD or self.infex.agents[i].quarantined:
                continue

            # enough tests?
            if cnt + 1 > tests_max:
                break
            cnt += 1

            # attetion: quarantined ones are not tested here again!
            # just realeased after 3 weeks
            test_result = self.probing.probe([self.infex.agents[i]])
            self.infex.agents[i].add_testresult(test_result)
            if test_result:
                self.infex.agents[i].quarantine(days=3*14)

        self.nb_tests += [(self.infex.t, cnt)]

class TestRandom(Test):
    """
    Test strategy:
    - Probability to test a person per day (including quarantined): <prob_test>
    - Probability to quarantine a person per day (including quarantined): <prob_quarantine>
    - Quarantine duration: <quarantine_days>
    """

    def __init__(self, probing: PoolableProbing, infex: InfectionExecution, **kwargs):
        super().__init__(probing, infex)
        self.kwargs = kwargs
        self.prob_test = kwargs['prob_test']
        self.prob_quarantine = kwargs['prob_quarantine']
        self.quarantine_days = kwargs['quarantine_days']

    def __str__(self):
        return \
f"""TestRandom:
probing={self.probing},
infex={self.infex},
prob_test={self.prob_test},
prob_quarantine={self.prob_quarantine},
quarantine_days={self.quarantine_days}
"""

    def do_test(self):
        tests_max = nb_available_tests(self.infex.t)
        cnt = 0

        pop = config.person_ids
        random.shuffle(pop)
        for i in pop:
            if self.infex.agents[i].symptoms is Symptoms.DEAD:
                # do not test dead
                continue
            
            # check if to test
            u = random.random()
            if u <= self.prob_test:
                # potentially test
                want_to_test = True
            else:
                want_to_test = False

            # check if to quarantine
            u = random.random()
            if u <= self.prob_quarantine:
                # quarantine
                self.infex.agents[i].quarantine(days=self.quarantine_days)

            # enough tests?
            if cnt + 1 > tests_max:
                #no
                break

            # else, make test if planned
            if want_to_test:
                cnt += 1
                # make the tests
                test_result = self.probing.probe([self.infex.agents[i]])
                self.infex.agents[i].add_testresult(test_result)

        self.nb_tests += [(self.infex.t, cnt)]


class TestSpreadingPredictionNN(Test):
    """
    Test strategy:
    - NN filename: <nn_filename>
    - Tests per day: <test_per_day>
    - Quarantine duration: <quarantine_days>
    - cutoff probability: <cutoff_prob>
    """
    def __init__(self, probing: PoolableProbing, infex: InfectionExecution, **kwargs):
        super().__init__(probing, infex)
        self.kwargs = kwargs
        self.nn_filename = kwargs['nn_filename']
        self.test_per_day = kwargs['test_per_day']
        self.quarantine_days = kwargs['quarantine_days']
        self.cutoff_prob = kwargs['cutoff_prob']
        # load NN
        self.model = learn_simple.load_NN(self.nn_filename, stateful=True)
        # state vector of all agents
        self.state_vec = []
        # prediction vector of all agents
        self.prediction_vec = []

    def __str__(self):
        return \
f"""TestSpreadingPredictionNN:
probing={self.probing},
infex={self.infex},
nn_filename={self.nn_filename},
test_per_day={self.test_per_day},
quarantine_days={self.quarantine_days},
cutoff_prob={self.cutoff_prob}
"""

    def __get_most_likely_spreading_ids__(self):
        # get [ (id, prediction) , ... ]
        #print(self.prediction_vec[-1])
        id_pred_vec = enumerate( self.prediction_vec[-1] )
        #print(list(id_pred_vec))
        # sort (decending) and filter
        ids = [ agent_id for agent_id, pred_prob in sorted( id_pred_vec, key=lambda x: x[1], reverse=True ) if pred_prob >= self.cutoff_prob ]        
        return ids

    def __update_state__(self):
        # updates the state that is required for the NN
        self.state_vec += self.infex.get_state_vec()
        #current_T = len(self.state_vec)
        # get properties
        symptom_none, symptom_mild, symptom_severe, symptom_dead, \
           category_MED, category_NUR, category_ADM, category_PAT, \
           infected, spreading, \
           working, quarantined, \
           contactgraphs, \
           test_positive, test_negative = learn_simple.get_properties(self.state_vec, self.infex)
        # prediction for all agents
        all_pred = None
        for i in range(config.N):
            # bring agent i into position 0
            learn_simple.swap_agents(i, 0, contactgraphs,
                symptom_none,
                symptom_mild,
                symptom_severe,
                symptom_dead,
                category_MED,
                category_NUR,
                category_ADM,
                category_PAT,
                infected,
                spreading,
                working,
                quarantined,
                test_positive,
                test_negative)
            # build input
            x = np.concatenate( (symptom_none, symptom_mild, symptom_severe, symptom_dead, \
                    category_MED, category_NUR, category_ADM, category_PAT, \
                    working, quarantined, test_positive, test_negative, \
                    learn_simple.contactgraphs2matrix(contactgraphs)), axis=1 )
            T_data = [ x[-1] ]
            # DATA_SIZE is 1 since has only 1 run to predict
            T_data = np.array(T_data).reshape(1, 1, learn_simple.OBS_LENGTH)
            # predict agent, stateful=True -> remember state for next batch
            pred = self.model.predict(T_data, verbose=0)
            pred = pred.reshape(1, 1)
            # add to all predictions
            if all_pred is None:
                all_pred = pred
            else:
                all_pred = np.concatenate((all_pred, pred), axis=1)
            # swap back (to be sure)
            learn_simple.swap_agents(i, 0, contactgraphs,
                symptom_none,
                symptom_mild,
                symptom_severe,
                symptom_dead,
                category_MED,
                category_NUR,
                category_ADM,
                category_PAT,
                infected,
                spreading,
                working,
                quarantined,
                test_positive,
                test_negative)
        # update predeiction vector
        self.prediction_vec = all_pred


    def do_test(self):
        # update prediction by NN
        self.__update_state__()
        # JUST FOR TEST!!!!!!
        #if self.infex.t == 1:
        #    self.infex.agents[0].quarantine(days=self.quarantine_days)
        #return
        # END JUST FOR TEST!!!
        # potentially do tests
        tests_max = min( nb_available_tests(self.infex.t), self.test_per_day )
        cnt = 0

        ids_to_test = self.__get_most_likely_spreading_ids__()
        print(f"Day {self.infex.t}: predict as likely infected: {ids_to_test}")
        for i in ids_to_test:
            # already quarantined?
            if self.infex.agents[i].quarantined:
                # take the next agent
                continue

            # enough tests?
            if cnt + 1 > tests_max:
                #no
                break

            # yes -> make the tests
            cnt += 1
            print(f"  testing {i}")
            test_result = self.probing.probe([self.infex.agents[i]])
            self.infex.agents[i].add_testresult(test_result)

            if test_result:
                # positive test
                print(f"  positive test for agent {i}")
                self.infex.agents[i].quarantine(days=self.quarantine_days)

        self.nb_tests += [(self.infex.t, cnt)]



class TestPool(Test):
    """
    Test strategy:
    - Test in pools.
    - Quarantine positive pools and retest shortly after individually.
    - Positively tested individuals are quarantined for longer.

    Arguments:
    individual_quarantine_days (=2*7) -- days a positive individual is quarantined (int)
    group_quarantine_days (=2) -- days a positive group is quarantined (int)
    poolsize (=10) -- pool size (int)
    k_group (=1) -- number of tests to test a group (ANDed tests) (! change to OR)
    days_if_positive (=7) -- number of days in which to test again after a positive test
                             in quarantine (int)
    k_release (=2) -- number of tests to test before starting release day counter
                      (ANDed tests) (int)
    days_after_negative (=14) -- release day counter: release after these days (int)
    """

    def __init__(self, probing: PoolableProbing, infex: InfectionExecution, **kwargs):
        super().__init__(probing, infex)
        self.kwargs = kwargs
        self.individual_quarantine_days = kwargs['individual_quarantine_days']
        #
        self.group_quarantine_days = kwargs['group_quarantine_days']
        self.poolsize = kwargs['poolsize']
        self.k_group = kwargs['k_group']
        #
        self.days_if_positive = kwargs['days_if_positive']
        self.k_release = kwargs['k_release']
        self.days_after_negative = kwargs['days_after_negative']
        #
        self.t = 0
        self.queue = []

    def __str__(self):
        return \
            f"""TestPool:
probing={self.probing},
kwargs={self.kwargs}"""

    def __group__(self, mylist: list, groupsize: int):
        """
        Returns: Groups <mylist> into <groupsize> sublists.
                 Last sublist may be shorter.
        """
        args = [iter(mylist)] * groupsize
        tmp = zip_longest(*args, fillvalue=None)
        # remove None from lists in groups
        tmp = list(map(lambda li: list(filter(None, li)),
                       tmp))
        # remove [] from groups
        return list(filter(lambda li: not(li == []),
                           tmp))

    def __schedule_quarantine_test__(self, agent: Agent, in_days: int, days_if_positive: int, k_release: int, days_after_negative: int):
        """
        Schedule a quarantine test for later.
        """
        # print(f'schedule test for {agent.id}')
        # remove old plans for this id
        self.queue = list(filter(lambda p: not(p['id'] == agent.id),
                                 self.queue))
        # add new plan for this id
        self.queue += [{'t': self.t + in_days,
                        'id': agent.id,
                        'days_if_positive': days_if_positive,
                        'k_release': k_release,
                        'days_after_negative': days_after_negative}]
        # print(f'schedule={self.queue}')

    def __schedule__(self, t: int):
        """
        Make schedule step.
        Curently just update time.
        """
        self.t = t

    def __get_scheduled__(self):
        """
        Get all test orders that are scheduled for today as a list.
        """
        # get scheduled agent ids
        scheduled = [sch for sch in self.queue if sch['t'] <= self.t]
        # cleanup queue
        self.queue = [sch for sch in self.queue if sch['t'] > self.t]
        return scheduled

    def __quarantine_ids__(self, ids: list, days: int, days_if_positive: int, k_release: int, days_after_negative: int):
        """
        Quarantine all ids.
        Further, schedule tests for them later on.
        """
        # print(f'quarantine_ids(ids= {ids}, ...)')
        for i in ids:
            # send to quarantine
            self.infex.agents[i].quarantine(days)
            # schedule test for day before quarantine stop
            self.__schedule_quarantine_test__(agent=self.infex.agents[i],
                                              in_days=days - 1,
                                              days_if_positive=days_if_positive,
                                              k_release=k_release,
                                              days_after_negative=days_after_negative)

    def do_test(self):
        # print(f'time={t}')
        tests_max = nb_available_tests(self.infex.t)
        cnt = 0
        # run quarantine test scheduler
        self.__schedule__(t)
        # get alive population
        alive_pop = self.__get_alive__()
        # get all non-quarantined
        nonquarantined_pop = list(filter(lambda i: not(self.infex.agents[i].quarantined),
                                         alive_pop))
        # remaining severe ill easily seen -> quarantine
        immediate_quarantine = list(filter(lambda i: self.infex.agents[i].symptoms is Symptoms.SEVERE,
                                           nonquarantined_pop))
        # print(f'immediate_quarantine={immediate_quarantine}')
        for i in immediate_quarantine:
            # Quarantine (long period).
            # If then positive (test self.k_release times) then quarantine again for
            # self.days_if_positive days.
            # If negative quarantine for self.days_after_negative days.
            self.__quarantine_ids__([i],
                                    days=self.individual_quarantine_days,
                                    days_if_positive=self.days_if_positive,
                                    k_release=self.k_release,
                                    days_after_negative=self.days_after_negative)
        # remove these
        nonquarantined_pop = list(
            set(nonquarantined_pop) - set(immediate_quarantine))
        # shuffe rest
        random.shuffle(nonquarantined_pop)
        # group them
        groups = self.__group__(nonquarantined_pop, self.poolsize)
        # print(f'groups={groups}')

        # go over groups
        for g in groups:
            # check if still tests available
            if cnt + self.k_group > tests_max:
                break
            cnt += self.k_group
            # test group
            test_result = self.probing.probe([self.infex.agents[i] for i in g], k=self.k_group)
            # TODO: add prool result with somethg like self.infex.agents[i].add_testresult(test_result)
            if test_result:
                # print(f'positive group={g}')
                # Quarantine all in positive group (short period).
                # If then positive (test self.k_release times) then quarantine again for
                # self.days_if_positive days.
                # If negative quarantine for self.days_after_negative days.
                self.__quarantine_ids__(g,
                                        days=self.group_quarantine_days,
                                        days_if_positive=self.days_if_positive,
                                        k_release=self.k_release,
                                        days_after_negative=self.days_after_negative)

        # get test plans for today
        scheduled_quarantined = self.__get_scheduled__()
        # go over scheduled quarantined ones, one by one
        # print(f'scheduled_quarantined={scheduled_quarantined}')
        for sch in scheduled_quarantined:

            if sch['k_release'] == 0:
                # no test needed, just release
                continue

            # check if still tests available
            if cnt + sch['k_release'] > tests_max:
                break
            cnt += sch['k_release']

            # test individual
            i = sch['id']
            test_result = self.probing.probe([self.infex.agents[i]], k=sch['k_release'])
            self.infex.agents[i].add_testresult(test_result)
            if test_result:
                # positive test
                # quarantine according to days and k_release in the schedule
                self.__quarantine_ids__([i],
                                        days=sch['days_if_positive'],
                                        days_if_positive=sch['days_if_positive'],
                                        k_release=sch['k_release'],
                                        days_after_negative=sch['days_after_negative'])
            else:
                # negative test
                if sch['days_after_negative'] > 0:
                    # keep them for another sch['days_after_negative'] days
                    self.__quarantine_ids__([i],
                                            days=sch['days_after_negative'],
                                            days_if_positive=None,
                                            k_release=0,  # this means release without test
                                            days_after_negative=None)

        self.nb_tests += [(self.infex.t, cnt)]


class TestContacts(TestPool):
    """
    On positive test or severe ill, quarantine contacts.
    Let them back only if they are negative.

    * Example Austrian hospital strategy

    individual_quarantine_days= 2*7
    contact_quarantine_days= 2*7     # contact of positives are quarantined for 14 days AND no symptoms
    lookback_days= 2                 # 2 days before symptom onset
    #
    contacts_check= False            # don't check these right away
    contacts_poolsize= 5,            #
        contacts_k_group= 1,             #
        #
    working_check= False             # don't check all others-> ignore next
    working_quarantine_days= 2*7     #
    working_poolsize= 5,             #
        working_k_group= 1,              #
        #
    days_if_positive= 7,             # if positive in quarantine, retest in 7 days (check!)
    k_release= 2                     # 2 negative tests required before release
    days_after_negative= 14          # after tests are negative, wait another 14 days (check!)

    """

    def __init__(self, probing: PoolableProbing, infex: InfectionExecution, **kwargs):
        # init TestPool
        super().__init__(probing, infex,
                         individual_quarantine_days=kwargs['individual_quarantine_days'],
                         group_quarantine_days=kwargs['working_quarantine_days'],
                         poolsize=kwargs['working_poolsize'],
                         k_group=kwargs['working_k_group'],
                         days_if_positive=kwargs['days_if_positive'],
                         k_release=kwargs['k_release'],
                         days_after_negative=kwargs['days_after_negative'])
        # init remaining
        self.kwargs = kwargs
        self.contact_quarantine_days = kwargs['contact_quarantine_days']
        self.lookback_days = kwargs['lookback_days']
        #
        self.contacts_check = kwargs['contacts_check']
        self.contacts_poolsize = kwargs['contacts_poolsize']
        self.contacts_k_group = kwargs['contacts_k_group']
        #
        self.working_check = kwargs['working_check']
        #
        self.lastday_severe_symptomatic = []

    def __str__(self):
        return \
            f"""TestContacts:
probing={self.probing},
kwargs={self.kwargs}"""

    def __quarantine_contacts__(self, t: int, myids: list):
        """
        Quarantines contacts of all in myids.
        Looks back self.lookback_days days for contacts.

        Depending on self.contacts_check check these with a pool
        """
        # print(f'asking for contacts={myids}')
        # get contacts
        contacts_ids = []
        for i in myids:
            contacts_ids += self.infex.get_contacts_fromtime(
                agent_id=i, t_from=t - self.lookback_days)
        # print(f'contacts_ids={contacts_ids}')

        if self.contacts_check:
            # not implemented yet
            raise NotImplementedError
            # TODO: quarantine and schedule for pool testing

        else:
            # just quarantine contacts (k= 0)
            # but only (!) if they are not quarantined yet,
            # otherwise let their quarantines and plans as they are (check!)
            contacts_ids = list(filter(lambda i: not(self.infex.agents[i].quarantined),
                                       contacts_ids))
            self.__quarantine_ids__(contacts_ids,
                                    days=self.contact_quarantine_days,
                                    days_if_positive=None,
                                    k_release=0,  # dont test again
                                    days_after_negative=None)

    @property
    def __new_severe_symptomatic__(self):
        # all alive ids
        alive_pop = self.__get_alive__()
        today_severe_symptomatic = list(filter(lambda i: self.infex.agents[i].symptoms is Symptoms.SEVERE,
                                               alive_pop))
        new_severe_symptomatic = list(
            set(today_severe_symptomatic) - set(self.lastday_severe_symptomatic))
        # update
        self.lastday_severe_symptomatic = today_severe_symptomatic
        # print(new_severe_symptomatic)
        return new_severe_symptomatic

    def do_test(self):
        tests_max = nb_available_tests(self.infex.t)
        cnt = 0
        # run quarantine test scheduler
        self.__schedule__(t)
        # get new severe ill
        immediate_quarantine = self.__new_severe_symptomatic__
        for i in immediate_quarantine:
            # quarantine (long period)
            # if then positive (test self.k_release times) then quarantine again
            # for self.days_if_positive days
            self.__quarantine_ids__([i],
                                    days=self.individual_quarantine_days,
                                    days_if_positive=self.days_if_positive,
                                    k_release=self.k_release,
                                    days_after_negative=self.days_after_negative)
        # quarantine and schedule for testing contacts
        # note: if they are alredy quarantined this leaves them as they are
        if len(immediate_quarantine) > 0:
            self.__quarantine_contacts__(t=t, myids=immediate_quarantine)

        if self.working_check:
            raise NotImplementedError
            # # overall population
            # alive_pop = self.__get_alive__(mystate)
            # # filter all quarantined
            # pop = list(filter( lambda i: not(mystate[i].quarantined),
            #                    alive_pop))
            # # shuffe them
            # random.shuffle(pop)
            # # group them
            # groups = self.__group__(pop, self.poolsize)
            # # print(groups)

            # # go over groups
            # for g in groups:
            #     # check if still tests available
            #     if cnt + self.k_group > tests_max:
            #         break
            #     cnt += self.k_group
            #     # test group
            #     test_result = self.probing.probe(mystate, g, k= self.k_group)
            #     if test_result:
            #         # quarantine all (short period)
            #         # if then positive (test 2 times) then quarantine again for 7 days
            #         self.__quarantine_ids__(mystate, g,
            #                                 days= self.group_quarantine_days,
            #                                 days_if_positive= self.days_if_positive,
            #                                 k_release= self.k_release,
            #                                 days_after_negative= self.days_after_negative)

        # get test plans for today
        scheduled_quarantined = self.__get_scheduled__()
        # go over scheduled quarantined plans ones, one by one
        to_quarantine = []
        for sch in scheduled_quarantined:

            if sch['k_release'] == 0:
                # dont test again, just release
                continue

            positive = False
            # first look
            i = sch['id']
            if self.infex.agents[i].symptoms is Symptoms.SEVERE:
                positive = True
            else:
                # check if still tests available
                if cnt + sch['k_release'] > tests_max:
                    break
                cnt += sch['k_release']
                # test individual
                positive = self.probing.probe([self.infex.agents[i]], k=sch['k_release'])
                self.infex.agents[i].add_testresult(positive)
            # act on result
            if positive:
                # positive test
                to_quarantine += [i]
                # quarantine according to days and k in the schedule
                self.__quarantine_ids__([i],
                                        days=sch['days_if_positive'],
                                        days_if_positive=sch['days_if_positive'],
                                        k_release=sch['k_release'],
                                        days_after_negative=sch['days_after_negative'])
            else:
                # negative test
                if sch['days_after_negative'] > 0:
                    # keep them for another sch['days_after_negative'] days
                    self.__quarantine_ids__([i],
                                            days=sch['days_after_negative'],
                                            days_if_positive=None,
                                            k_release=0,  # dont test again
                                            days_after_negative=None)

        self.nb_tests += [(self.infex.t, cnt)]


def get_Test(probing: PoolableProbing, infex: InfectionExecution, usetest: dict):
    if usetest['type'] in REGISTERED_TESTS:
        # print("Create test...")
        return eval( f"{usetest['type']}(probing=probing, infex=infex, **usetest['parameters'])" )
    else:
        assert(False)
