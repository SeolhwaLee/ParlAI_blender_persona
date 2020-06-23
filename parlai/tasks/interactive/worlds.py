#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from copy import deepcopy

from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.core.message import Message


class InteractiveWorld(DialogPartnerWorld):
    """
    Simple interactive world involving just two agents talking.

    In more sophisticated worlds the environment could supply information, e.g. in
    tasks/convai2 both agents are given personas, so a world class should be written
    especially for those cases for given tasks.
    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.init_contexts(shared=shared)
        self.turn_cnt = 0

    def init_contexts(self, shared=None):
        """
        Override to load or instantiate contexts to be used to seed the chat.
        """
        pass

    def get_contexts(self):
        """
        Override to return a pair of contexts with which to seed the episode.

        This function will be called before the first turn of every episode.
        """
        return ['', '']

    def finalize_episode(self):
        print("CHAT DONE ")
        if not self.epoch_done():
            print("\n... preparing new chat... \n")

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        if self.turn_cnt == 0:
            self.p1, self.p2 = self.get_contexts()

        acts = self.acts
        agents = self.agents
        if self.turn_cnt == 0 and self.p1 != '':
            # add the context on to the first message to agent 0
            context_act = Message(
                {'id': 'context', 'text': self.p1, 'episode_done': False}
            )
            agents[0].observe(validate(context_act))
        try:
            act = deepcopy(agents[0].act()) # {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': False, 'text': 'hey'}

            print("test", act)
        except StopIteration:
            self.reset()
            self.finalize_episode()
            self.turn_cnt = 0
            return
        acts[0] = act
        # print("test", acts[0]) # {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': False, 'text': 'hi'}
        if self.turn_cnt == 0 and self.p2 != '':
            # add the context on to the first message to agent 1
            context_act = Message(
                {'id': 'context', 'text': self.p2, 'episode_done': False}
            )
            agents[1].observe(validate(context_act))
        agents[1].observe(validate(act))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()
        self.turn_cnt += 1

        if act['episode_done']:
            self.finalize_episode()
            self.turn_cnt = 0

    def parley_persona_script(self, input_path, output_path, model_name, multi_check, turn_n):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        script_input_path = str(input_path)
        script_file = open(script_input_path, 'r', encoding='utf-8')

        script_out_path = str(output_path)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = script_input_path.split('/')[-1].split('.')[0]

        if model_name.find(":") != -1:
            model_name = model_name.split(':')[-1]
        else:
            model_name = model_name.split('/')[-1]

        if model_name.find('blender') != -1:
            script_response = open(script_out_path + '/' + file_name + '_' + model_name.split('/')[-2] + '_' + timestr +
                                   '.txt', 'w')
        else:
            script_response = open(script_out_path + '/' + file_name + '_' + model_name + '_' + timestr +
                                   '.txt', 'w')

        count = 0
        for raw_text in script_file:
            count += 1
            raw_text = raw_text.replace('\n', '')

            if multi_check == True:
                if turn_n == 2:
                    turn1 = raw_text.split('</s>')[0]
                    turn2 = raw_text.split('</s>')[1]
                    turn_temp = [turn1, turn2]

                    for index, turn_each in enumerate(turn_temp):
                        if index == 1:
                            # second turn
                            # acts[0] = {'id': 'localHuman', 'episode_done': False, 'label_candidates': None,
                            #            'text': str(turn_each)}

                            try:
                                # act = deepcopy(agents[0].act())
                                act = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': False,
                                           'text': str(turn_each)}
                            except StopIteration:
                                self.reset()
                                self.finalize_episode()
                                self.turn_cnt = 0
                                return

                            acts[0] = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': False,
                                       'text': str(turn_each)}
                            agents[1].observe(validate(acts[0]))
                            acts[1] = agents[1].act()
                            agents[0].observe(validate(acts[1]))

                            result = acts[1]['text']
                            script_response.write("%s\n" % (result))
                            self.update_counters()

                        # first turn
                        # acts[0] = {'id': 'localHuman', 'episode_done': True, 'label_candidates': None,
                        #            'text': str(turn_each)}
                        try:
                            # act = deepcopy(agents[0].act())
                            act = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': True,
                                       'text': str(turn_each)}
                        except StopIteration:
                            self.reset()
                            self.finalize_episode()
                            self.turn_cnt = 0
                            return

                        acts[0] = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': True,
                                   'text': str(turn_each)}
                        agents[1].observe(validate(acts[0]))
                        acts[1] = agents[1].act()
                        agents[0].observe(validate(acts[1]))

                        result = acts[1]['text']
                        # script_response.write("%s\n" % (result))
                        self.update_counters()

                    turn_temp = []

                elif turn_n == 3:
                    turn1 = raw_text.split('</s>')[0]
                    turn2 = raw_text.split('</s>')[1]
                    turn3 = raw_text.split('</s>')[2]
                    # turn2 = raw_text.split('</s>')[1].split('<\s>')[0]
                    # turn3 = raw_text.split('<\s>')[1]
                    # if turn2.find('</s>') != -1:
                    #     turn3 = raw_text.split('</s>')[2]
                    # elif raw_text.find('<\s>') != -1:
                    #     turn3 = raw_text.split('<\s>')[1]
                    # else:
                    #     turn3 = ''
                    #     print("Check the turn3!!")
                    turn_temp = [turn1, turn2, turn3]

                    for index, turn_each in enumerate(turn_temp):
                        if index == 1:
                            # second turn
                            # acts[0] = {'id': 'localHuman', 'episode_done': False, 'label_candidates': None,
                            #            'text': str(turn_each)}
                            try:
                                # act = deepcopy(agents[0].act())
                                act = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': False,
                                           'text': str(turn_each)}
                            except StopIteration:
                                self.reset()
                                self.finalize_episode()
                                self.turn_cnt = 0
                                return

                            acts[0] = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': False,
                                       'text': str(turn_each)}
                            agents[1].observe(validate(acts[0]))
                            acts[1] = agents[1].act()
                            agents[0].observe(validate(acts[1]))

                            result = acts[1]['text']
                            # script_response.write("%s\n" % (result))
                            self.update_counters()

                        if index == 2:
                            # third turn
                            # acts[0] = {'id': 'localHuman', 'episode_done': False, 'label_candidates': None,
                            #            'text': str(turn_each)}
                            try:
                                # act = deepcopy(agents[0].act())
                                act = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': False,
                                           'text': str(turn_each)}
                            except StopIteration:
                                self.reset()
                                self.finalize_episode()
                                self.turn_cnt = 0
                                return
                            acts[0] = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': False,
                                       'text': str(turn_each)}
                            agents[1].observe(validate(acts[0]))
                            acts[1] = agents[1].act()
                            agents[0].observe(validate(acts[1]))

                            result = acts[1]['text']
                            script_response.write("%s\n" % (result))
                            self.update_counters()

                        # first turn
                        try:
                            # act = deepcopy(agents[0].act())
                            act = {'id': 'localHuman', 'episode_done': True, 'label_candidates': None,
                                       'text': str(turn_each)}
                        except StopIteration:
                            self.reset()
                            self.finalize_episode()
                            self.turn_cnt = 0
                            return
                        acts[0] = {'id': 'localHuman', 'episode_done': True, 'label_candidates': None,
                                   'text': str(turn_each)}
                        agents[1].observe(validate(acts[0]))
                        acts[1] = agents[1].act()
                        agents[0].observe(validate(acts[1]))

                        # result = acts[1]['text']
                        # script_response.write("%s\n" % (result))
                        self.update_counters()

                    turn_temp = []

            else:
                # acts[0] = {'id': 'localHuman', 'episode_done': True, 'label_candidates': None, 'text': str(raw_text)}
                # print(acts[0])
                if self.turn_cnt == 0:
                    self.p1, self.p2 = self.get_contexts()

                acts = self.acts
                agents = self.agents
                if self.turn_cnt == 0 and self.p1 != '':
                    # add the context on to the first message to agent 0
                    context_act = Message(
                        {'id': 'context', 'text': self.p1, 'episode_done': False}
                    )
                    agents[0].observe(validate(context_act))

                try:
                    # act = deepcopy(agents[0].act())
                    act = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': True,
                               'text': str(raw_text)}
                except StopIteration:
                    self.reset()
                    self.finalize_episode()
                    self.turn_cnt = 0
                    return

                acts[0] = {'id': 'safeLocalHuman', 'label_candidates': None, 'episode_done': True, 'text': str(raw_text)}
                # acts[0] = {'id': 'context', 'episode_done': True, 'label_candidates': None, 'text': str(raw_text)}

                if self.turn_cnt == 0 and self.p2 != '':
                    # add the context on to the first message to agent 1
                    context_act = Message(
                        {'id': 'context', 'text': self.p2, 'episode_done': False}
                    )
                    agents[1].observe(validate(context_act))

                agents[1].observe(validate(act))
                acts[1] = agents[1].act()
                agents[0].observe(validate(acts[1]))
                result = acts[1]['text']
                script_response.write("%s\n" % (result))
                self.update_counters()
                self.turn_cnt += 1

                if act['episode_done']:
                    self.finalize_episode()
                    self.turn_cnt = 0

        script_response.close()
        print("script response complete!")
        import sys
        sys.exit()