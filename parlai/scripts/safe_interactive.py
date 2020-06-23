#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script for safety protected interaction between a local human keyboard input and a
trained model.
"""

from parlai.core.params import ParlaiParser
from parlai.scripts.script import ParlaiScript
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.safe_local_human.safe_local_human import SafeLocalHumanAgent
import parlai.utils.logging as logging
import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Interactive chat with a model')
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--display-prettify',
        type='bool',
        default=False,
        help='Set to use a prettytable when displaying '
        'examples with text candidates',
    )
    parser.add_argument(
        '--display-ignore-fields',
        type=str,
        default='label_candidates,text_candidates',
        help='Do not display these fields',
    )
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    parser.add_argument(
        '-sc',
        '--script-chateval',
        type='bool',
        default=False,
        dest='chat_script',
        help='Chateval script read file'
             'True: chateval evaluation, False: single-turn conversation with agent(original model)',
    )
    parser.add_argument(
        '-scip',
        '--chateval-input-path',
        type=str,
        default=None,
        dest='script_input_path',
        help='Chateval script input path',
    )
    parser.add_argument(
        '-scop',
        '--chateval-output-path',
        type=str,
        default=None,
        dest='script_output_path',
        help='Chateval result output path',
    )
    parser.add_argument(
        '--chateval-multi-num',
        type=int,
        default=0,
        dest='chateval_multi_num',
        help='True is chateval multiturn setting, turn coverage count.',
    )
    parser.add_argument(
        '--chateval-multi',
        type='bool',
        default=False,
        hidden=True,
        dest='chateval_multi',
        help='True is chateval multiturn setting, False just single turn.',
    )
    parser.set_defaults(interactive_mode=True, task='interactive')
    SafeLocalHumanAgent.add_cmdline_args(parser)
    return parser


def safe_interactive(opt, print_parser=None):
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()
    human_agent = SafeLocalHumanAgent(opt)
    world = create_task(opt, [human_agent, agent])

    # Interact until episode done
    # while True:
    #     world.parley()
    #     bot_act = world.get_acts()[-1]
    #     if 'bot_offensive' in bot_act and bot_act['bot_offensive']:
    #         agent.reset()
    #
    #     if opt.get('display_examples'):
    #         print('---')
    #         print(world.display())
    #     if world.epoch_done():
    #         logging.info('epoch done')
    #         break
    if not opt.get('chat_script'):
        '''for chateval script evaluation'''
        # Interact until episode done
        while True:
            world.parley()
            bot_act = world.get_acts()[-1]
            if 'bot_offensive' in bot_act and bot_act['bot_offensive']:
                agent.reset()

            if opt.get('display_examples'):
                print('---')
                print(world.display())
            if world.epoch_done():
                logging.info('epoch done')
                break
    # else:
    elif opt.get('chat_script') and opt.get('include_personas'):
        while True:
            # world.parley_script(opt.get('script_input_path'), opt.get('script_output_path'), opt.get('model-file'))
            # turn_available = [0, 2, 3]
            world.parley_persona_script(opt.get('script_input_path'), opt.get('script_output_path'), opt.get('model_file'),
                                opt.get('chateval_multi'),opt.get('chateval_multi_num'))
            bot_act = world.get_acts()[-1]
            if 'bot_offensive' in bot_act and bot_act['bot_offensive']:
                agent.reset()

            if opt.get('display_examples'):
                print("---")
                print(world.display())
            if world.epoch_done():
                logging.info('epoch done')
                break

    else:
        while True:
            # world.parley_script(opt.get('script_input_path'), opt.get('script_output_path'), opt.get('model-file'))
            # turn_available = [0, 2, 3]
            world.parley_script(opt.get('script_input_path'), opt.get('script_output_path'), opt.get('model_file'),
                                opt.get('chateval_multi'),opt.get('chateval_multi_num'))
            bot_act = world.get_acts()[-1]
            if 'bot_offensive' in bot_act and bot_act['bot_offensive']:
                agent.reset()

            if opt.get('display_examples'):
                print("---")
                print(world.display())
            if world.epoch_done():
                logging.info('epoch done')
                break


class SafeInteractive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return safe_interactive(self.opt, print_parser=self.parser)


if __name__ == '__main__':
    random.seed(42)
    SafeInteractive.main()
