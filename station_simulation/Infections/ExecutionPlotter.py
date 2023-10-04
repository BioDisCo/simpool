import numpy as np
import matplotlib.pyplot as plt

from Infections.InfectionExecution import InfectionExecution
from Infections.Agent import Agent, Symptoms

COLOR_POS_TEST = (0.6, 0.0, 0.0)
COLOR_NEG_TEST = (0.0, 0.6, 0.0)


class ExecutionPlotter(object):
    def __init__(self, N: int, T: int):
        self.N = N
        self.T = T
        self.agents_order: list = []
        self.agents_states: dict = {} # [id][time] -> state 

    def __state_to_color__(self, agent_state: dict) -> list:
        # rna as red
        if agent_state['rna'] > 0:
            # cutoff to still see levels
            cutoff = max(0.2, agent_state['rna'])
            r,g,b = (1.0,1.0-cutoff,1.0-cutoff)
        else:
            r,g,b = (1,1,1)
        return [r,g,b]

    def __get_firstdayofspreading__(self, agent_id) -> float:
        spreading_days = [i for i, s in enumerate(self.agents_states[agent_id]) if s['rna'] > 0]
        if len(spreading_days) > 0:
            return spreading_days[0]
        else:
            return float("inf")

    def __get_agent_img__(self, agent_id: int):
        agent_colors = []
        for s in self.agents_states[agent_id]:
            agent_colors += [self.__state_to_color__(s)]
        return agent_colors

    def __draw_infected_events__(self):
        for line, agent_id in enumerate(self.agents_order):
            infected_days = [i for i, s in enumerate(self.agents_states[agent_id]) if s['infected']]
            first_day = infected_days[0] if len(infected_days) > 0 else None
            if first_day is not None:
                print(f'{agent_id}: {first_day}')
                plt.plot([first_day], [line], 'ko', markeredgewidth=0.0)

    def __draw_quarantined__(self):
        for line, agent_id in enumerate(self.agents_order):
            quarantined : bool = self.agents_states[agent_id][0]['quarantined']
            for t, s in enumerate(self.agents_states[agent_id]):
                if (t == 0) and s['quarantined']:
                    # quarantine event
                    plt.plot([t], [line], '>', color=(0,0.6,0), alpha=0.8, markeredgewidth=0.0)
                elif (not quarantined) and s['quarantined']:
                    # quarantine event
                    plt.plot([t], [line], '>', color=(0,0.6,0), alpha=0.8, markeredgewidth=0.0)
                    print(f"quarantined: day {t} agent {agent_id}")
                elif quarantined and (not s['quarantined']):
                    # dequarantine event
                    plt.plot([t], [line], '<', color=(0,0.6,0), alpha=0.8, markeredgewidth=0.0)
                quarantined = s['quarantined']

    def __draw_test__(self):
        for line, agent_id in enumerate(self.agents_order):
            for t, s in enumerate(self.agents_states[agent_id]):
                if s['testresult'] is not None:
                    # test event
                    if s['testresult']:
                        # pos test
                        plt.plot([t], [line], '*', color=COLOR_POS_TEST, alpha=0.8, markeredgewidth=0.0)
                    else:
                        # neg test
                        plt.plot([t], [line], '*', color=COLOR_NEG_TEST, alpha=0.8, markeredgewidth=0.0)

    def __draw_symptoms_events__(self):
        for line, agent_id in enumerate(self.agents_order):
            mild_days = [i for i, s in enumerate(self.agents_states[agent_id]) if s['symptoms'] is Symptoms.MILD]
            severe_days = [i for i, s in enumerate(self.agents_states[agent_id]) if s['symptoms'] is Symptoms.SEVERE]
            first_day_mild = mild_days[0] if len(mild_days) > 0 else None
            first_day_severe = severe_days[0] if len(severe_days) > 0 else None
            if first_day_mild is not None:
                plt.plot([first_day_mild], [line], 'o', color=(0.3,0.3,1.0), markeredgewidth=0.0)
                last_day_mild = mild_days[-1] if mild_days[-1] < self.T else None
                if last_day_mild is not None:
                    plt.plot([last_day_mild], [line], 'o', color=(0.3,0.3,1.0), markeredgewidth=0.0)
            if first_day_severe is not None:
                plt.plot([first_day_severe], [line], 'o', color=(0,0,0.4), markeredgewidth=0.0)
                last_day_severe = severe_days[-1] if severe_days[-1] < self.T else None
                if last_day_severe is not None:
                    plt.plot([last_day_severe], [line], 'o', color=(0,0,0.4), markeredgewidth=0.0)
            

    @property
    def image(self):
        img = np.ones((self.N,self.T,3))
        for line, i in enumerate(self.agents_order):
            img[line][:] = self.__get_agent_img__(agent_id=i)
        return img

    def add_agent(self, infex: InfectionExecution, agent_id: int):
        self.agents_states[agent_id] = [ infex.states[t][agent_id] for t in range(self.T) ]
        self.agents_order += [agent_id]

    def sort(self):
        self.agents_order.sort(key=lambda i : self.__get_firstdayofspreading__(agent_id=i))

    def plot(self, title: str, save: bool = False):
        plt.figure()
        ax = plt.subplot(111)
        ax.set_xlabel('day')
        ax.set_ylabel('person')
        ax.set_title(f'Infection progress and quarantine: {title}')
        plt.imshow(self.image)
        self.__draw_infected_events__()
        self.__draw_symptoms_events__()
        self.__draw_quarantined__()
        self.__draw_test__()
        if save:
            plt.savefig(f'fig_Infection_progress_and_quarantine_{title}.pdf')
