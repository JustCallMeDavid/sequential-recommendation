import torch
from tqdm import tqdm
import logging

from models import NextItNet, Caser, GRU4Rec, SAS4Rec, Random4Rec
from preprocessing import process_rr, process_rs, generate_splits
from transition_store import TransitionStore
from metrics import MetricsHelper

_logger = logging.getLogger('sequential-recommendation')


def _compute_matching_items_batch(pred, tgt, batch_size):
    """computes the number of items in a batch that match the target items, i.e., the amount of correct predictions"""
    return sum(pred.argmax(dim=1) == tgt) / batch_size


def train(train_store, model, optimizer, criterion, args, device, val_store=None):

    if type(model) == Random4Rec:
        return  # no training required for random model

    model.to(device)

    running_loss = 0
    running_acc = 0
    pbar = tqdm(range(args.num_epochs), leave=False)

    for e in pbar:
        if args.validation_interval is not None and e % args.validation_interval == 0 and val_store is not None:
            hrs, ndcgs = evaluate(model, val_store, args, device)
            print(f'\nIn Epoch {e}: Validation results are:\nHit-Rates (click, buy) {hrs}\nNDCGs (click, buy) {ndcgs}')

        model.train()
        for batch in train_store.dataloader:
            st, lst, a, buy, stp1, lstp1, done = batch
            st = st.to(device)
            a = a.to(device)

            if type(model) == Caser:
                out = model(st)
            elif type(model) == NextItNet or type(model) == GRU4Rec or type(model) == SAS4Rec:
                out = model(st, lst)
            else:
                raise NotImplementedError()

            loss = criterion(out, a)  # the actions taken by the user are the target
            acc = _compute_matching_items_batch(out, a, args.batch_size)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss = running_loss * 0.99 + 0.01 * loss.item()
            running_acc = running_acc * 0.99 + 0.01 * acc
            pbar.set_description(f'Avg. Loss: {running_loss}, Avg. Acc.: {running_acc}')

    return model


def prepare_data(path, dtype):
    if dtype == 'retail-rocket' or dtype == 'rr':
        # retail-rocket processing
        _logger.log(msg=f'Processing retail-rocket dataset from {path}', level=logging.INFO)
        df, cnts = process_rr(path)
    elif dtype == 'recsys15' or dtype == 'recsys15':
        # recsys15 processing
        _logger.log(msg=f'Processing recsys15 dataset from {path}', level=logging.INFO)
        df, cnts = process_rs(path)
    else:
        raise ValueError(f'Data type passed {dtype} was not recognized.')

    return generate_splits(df, len(df.user_id.unique())), cnts


def evaluate(model, transition_store, args, device):

    model.eval()
    model.to(device)

    mh = MetricsHelper(args.top_k)  # metrics helper handles memory-efficient repeated metrics updates

    # NOTE: running evaluation on full dataset at once not feasible due to memory constraints -> aggregate by user
    for user_id in transition_store.get_userids():
        preds, actions, rewards, buys = list(), list(), list(), list()
        for batch in transition_store.get_user_dataloader(user_id):
            st, lst, a, buy, stp1, lstp1, done = batch
            st = st.to(device)
            buy = buy.to(device).byte()

            # generate predictions
            if type(model) == Caser or type(model) == Random4Rec:
                # sort the predictions (softmax-values) by size, take their index (== item_id)
                preds.append(torch.argsort(model(st), dim=1, descending=True))
            elif type(model) == NextItNet or type(model) == GRU4Rec or type(model) == SAS4Rec:
                preds.append(torch.argsort(model(st, lst), dim=1, descending=True))

            actions.append(a)
            buys.append(buy)
        # transform interim tensors to lists
        preds = torch.cat(preds, dim=0).cpu().numpy()
        actions = torch.cat(actions, dim=0).cpu().numpy()
        buys = torch.cat(buys, dim=0).cpu().numpy()
        # needs to be computed per user, storing all preds in history results in memory issues (over 60GB RAM on rr)
        mh.update_hr_ndcg(preds=preds, actions=actions, buys=buys)

    # scale by dividing through totals
    return mh.get_hr_ndcg()


def run(args):
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    if args.dev_mode:
        _logger.log(msg=f'Running in dev mode. Data loaded from {args.dev_mode_path}', level=logging.INFO)
        data_path = args.dev_mode_path
    else:
        _logger.log(msg=f'Data loaded from {args.path}', level=logging.INFO)
        data_path = args.path

    dfs, cnts = prepare_data(data_path, args.data_type)
    train_store, val_store, test_store = \
        TransitionStore(dfs[0], args.data_type, args.max_seq_len, args.batch_size, 'train', args.dev_mode), \
        TransitionStore(dfs[1], args.data_type, args.max_seq_len, args.batch_size, 'val', args.dev_mode), \
        TransitionStore(dfs[2], args.data_type, args.max_seq_len, args.batch_size, 'test', args.dev_mode)

    args.out_size = cnts[0] + 1  # out_size equals total number of items (+1 due to padding item)

    if args.model_type.lower() == 'nextitnet':
        model = NextItNet(args)
    elif args.model_type.lower() == 'caser':
        model = Caser(args)
    elif args.model_type.lower() == 'sas4rec':
        model = SAS4Rec(args)
    elif args.model_type.lower() == 'gru4rec':
        model = GRU4Rec(args)
    elif args.model_type.lower() == 'random':
        model = Random4Rec(args)
    else:
        raise ValueError(f'Model type {args.model_type} was not recognized.')

    _logger.log(msg=f'Model initialized as {model}', level=logging.INFO)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss()

    train(train_store, model, optimizer, criterion, args, device, val_store=val_store)

    # run test
    hrs, ndcgs = evaluate(model, test_store, args, device)
    print(f'\nTEST\nTest results are: Hit-Rates (click, buy) {hrs} and NDCGs (click, buy) {ndcgs}')
