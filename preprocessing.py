import random
import pandas as pd
import pathlib

rr_columns = ['ts', 'user_id', 'event', 'item_id', 'transaction_id']
rs_buy_columns = ['user_id', 'ts', 'item_id', 'price', 'quantity']
rs_click_columns = ['user_id', 'ts', 'item_id', 'category']

buys_path = pathlib.Path('yoochoose-buys.dat')
clicks_path = pathlib.Path('yoochoose-clicks.dat')


def _transform_event_codes(x):
    if x == 'view':
        return 0
    elif x == 'addtocart':
        return 1
    elif x == 'transaction':
        raise NotImplementedError('Transactions are currently not supported.')
    else:
        raise ValueError(f"Event code {x} not recognized.")


def process_rr(data_path):
    """processes the retail-rocket dataset from file"""

    df = pd.read_csv(data_path, names=rr_columns, skiprows=1)

    # drop transactions form the dataset, addtocart is treated as a purchase (to be coherent with recsys15)
    df = df[pd.isna(df.transaction_id)]  # drop transaction information from the dataset
    df.drop('transaction_id', axis=1, inplace=True)
    df.event = df.event.apply(lambda x: _transform_event_codes(x))
    return _process(df)


def process_rs(data_path):
    """processes the recsys15 dataset from file"""
    # load data from the two datafiles
    df_buys = pd.read_csv(data_path / buys_path, header=None, names=rs_buy_columns)
    df_clicks = pd.read_csv(data_path / clicks_path, header=None, names=rs_click_columns)
    # drop unnecessary columns
    df_buys.drop(['price', 'quantity'], axis=1, inplace=True)
    df_clicks.drop(['category'], axis=1, inplace=True)
    # encode types
    df_clicks['event'] = 0
    df_buys['event'] = 1
    # concat dataframe
    df = df_clicks.append(df_buys)
    assert len(df) == (len(df_clicks) + len(df_buys))  # check no rows lost
    return _process(df)


def _process(df, min_interactions_user=3, min_interactions_item=3):
    # Note: in marginal cases it is possible for a user/item to still have too few interactions
    user_cnts = df.user_id.value_counts()
    users_gtn = user_cnts[user_cnts >= min_interactions_user].index.to_list()
    df = df[df.user_id.isin(users_gtn)]
    item_cnts = df.item_id.value_counts()
    items_gtn = item_cnts[item_cnts >= min_interactions_item].index.to_list()
    df = df[df.item_id.isin(items_gtn)]

    # transforms item_id, user_id and event according to order
    unique_items = df.item_id.unique()
    # the +1 is needed to ensure that no item_id is 0, since this is the padding item
    item_mapping_dict = {k: v + 1 for v, k in enumerate(unique_items)}
    df.item_id = df.item_id.apply(lambda x: item_mapping_dict[x])
    assert 0 not in df.item_id.unique()  # make sure id 0 is not given, padding item
    total_items = len(unique_items)

    unique_users = df.user_id.unique()
    user_mapping_dict = {k: v for v, k in enumerate(unique_users)}
    df.user_id = df.user_id.apply(lambda x: user_mapping_dict[x])
    total_users = len(unique_users)

    df.sort_values(by=['user_id', 'ts'], inplace=True)

    return df, (total_items, total_users)  # , (item_mapping_dict, user_mapping_dict)


def generate_splits(data, total_users, train_pct=0.8, val_pct=0.1, test_pct=0.1):
    """splits the data by user_id into the specified ratios"""

    assert train_pct + val_pct + test_pct == 1  # use all data
    assert train_pct >= 0 and val_pct >= 0 and test_pct >= 0

    train_idx, val_idx = int(total_users * train_pct), int(total_users * val_pct)
    idx_lst = list(range(total_users))
    random.shuffle(idx_lst)
    train_ids, val_ids, test_ids = \
        idx_lst[0:train_idx], idx_lst[train_idx:train_idx + val_idx], idx_lst[train_idx + val_idx:]

    return data[data.user_id.isin(train_ids)], data[data.user_id.isin(val_ids)], data[data.user_id.isin(test_ids)]
