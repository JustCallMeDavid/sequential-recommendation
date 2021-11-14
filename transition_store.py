import copy
import pathlib
import ast
import pandas as pd
import torch.utils.data
from tqdm import tqdm
import logging

from constants import PADDING_ITEM

_logger = logging.getLogger('sequential-recommendation')


class TransitionStore:

    def __init__(self, data, data_type, max_seq_len, batch_size, use, dev_mode):
        assert use in ['train', 'test', 'val']
        assert data_type in ['recsys15', 'retail-rocket']

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.use = use
        self.transitions = None
        self.dev_mode = dev_mode
        self.data_type = data_type

        if self.dev_mode:
            self.file_path = pathlib.Path(
                f'data/{self.data_type}/transition_store/dev/transition_store_{self.use}.csv')  # store the transitions
        else:
            self.file_path = pathlib.Path(
                f'data/{self.data_type}/transition_store/transition_store_{self.use}.csv')

        if not self._load_transitions_from_file():
            self._generate_transition_store(data)

        if self.use == 'train':
            assert self.transitions is not None
            self.dataloader = self._create_dataloader(self.transitions)
        else:
            self._create_user_dataloader_dict()

    def __len__(self):
        return len(self.transitions)

    def _pad_history(self, lst):
        ret_lst = copy.deepcopy(lst)
        if len(ret_lst) >= self.max_seq_len:
            ret_lst = ret_lst[-self.max_seq_len:]
        else:
            ret_lst.extend([PADDING_ITEM] * (self.max_seq_len - len(lst)))
        return ret_lst

    def _get_state_length(self, st):
        return 1 if len(st) == 0 else min(self.max_seq_len, len(st))

    def _generate_transition_store(self, data):

        item_ids, user_ids = data.item_id.unique(), data.user_id.unique()
        user_groups = data.groupby('user_id')  # bundle data by user

        # store generated transition values
        st_lst, lst_lst, a_lst, buy_lst, stp1_lst, lstp1_lst, done_lst, uids_lst = [], [], [], [], [], [], [], []

        for id_ in tqdm(user_ids):
            current_user_data = user_groups.get_group(id_)
            # make sure the list is sorted by timestamp
            assert all(b >= a for a, b in zip(current_user_data.ts.to_list(), current_user_data.ts.to_list()[1:]))
            user_history = list()

            for _, row in current_user_data.iterrows():
                st = copy.deepcopy(user_history)
                lst_lst.append(self._get_state_length(st))
                st = self._pad_history(st)
                a_lst.append(row.item_id)
                buy_lst.append(row.event)
                st_lst.append(st)
                user_history.append(row.item_id)  # continue user interaction history with current item
                stp1 = copy.deepcopy(user_history)  # next state with current interacted with item
                lstp1_lst.append(self._get_state_length(stp1))
                stp1 = self._pad_history(stp1)
                stp1_lst.append(stp1)
                done_lst.append(False)
                uids_lst.append(row.user_id)
            done_lst[-1] = True  # interaction is done on the last iteration, set done to True (overwrite prev. false)

        # check all lengths equal
        assert len(st_lst) == len(lst_lst) == len(a_lst) == len(buy_lst) == len(stp1_lst) == len(lstp1_lst) \
               == len(done_lst) == len(uids_lst)

        # generate dataframe from lists
        transitions = pd.DataFrame(zip(st_lst, lst_lst, a_lst, buy_lst, stp1_lst, lstp1_lst, done_lst, uids_lst),
                                   columns=['st', 'lst', 'a', 'buy', 'stp1', 'lstp1', 'done', 'user_id'])
        self._save_transitions_to_file(transitions)
        self._load_transitions_from_file()

    def _save_transitions_to_file(self, transitions):
        _logger.log(msg=f'Transition store saved to {self.file_path}.', level=logging.INFO)
        transitions.to_csv(self.file_path, index=False)

    def _load_transitions_from_file(self):
        """Attempts to load the transitions from file. Returns True if successful else False."""
        try:
            transitions = pd.read_csv(self.file_path)
            _logger.log(msg=f'Transitions loaded from file at {self.file_path}.', level=logging.INFO)
        except FileNotFoundError:
            _logger.log(msg=f'No Transitions file found at {self.file_path}.', level=logging.INFO)
            return False

        self.transitions = transitions
        # convert lists inside dataframe from str to list objects
        self.transitions.st = self.transitions.st.apply(lambda x: ast.literal_eval(x))
        self.transitions.stp1 = self.transitions.stp1.apply(lambda x: ast.literal_eval(x))
        return True

    def _create_user_dataloader_dict(self):
        self.user_data = dict()
        user_grps = self.transitions.groupby('user_id')
        for id_ in self.transitions.user_id.unique():
            self.user_data[id_] = self._create_dataloader(user_grps.get_group(id_))

    def get_user_dataloader(self, user_id):
        assert self.user_data is not None
        return self.user_data[user_id]

    def _create_dataloader(self, data):
        assert self.transitions is not None

        dataset = torch.utils.data.TensorDataset(torch.tensor(data.st.tolist()),
                                                 torch.tensor(data.lst.tolist()),
                                                 torch.tensor(data.a.tolist()),
                                                 torch.tensor(data.buy.tolist()),
                                                 torch.tensor(data.stp1.tolist()),
                                                 torch.tensor(data.lstp1.tolist()),
                                                 torch.tensor(data.done.tolist()))
        # shuffle data (otherwise would not be i.i.d.)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True, drop_last=False)

    def get_userids(self):
        if self.user_data is not None:
            return self.user_data.keys()
        elif self.transitions is not None:
            return self.transitions.unique()
        else:
            raise RuntimeError('Neither transitions nor user data were initialized, cannot return user ids.')
