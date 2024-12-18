from settings import *


# helper routing to log a message with time
def log_message(label_string, write_=True):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%f')
    print("{}: {}".format(st, label_string))
    if write_:
        with open(log_process_name, "a") as f:
            f.write("{}: {}".format(st, label_string))
            f.write("\n")
    return


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None, chronological=True):
    df = df.sort_values(['Month'], ascending=True, inplace=False).reset_index(drop=True, inplace=False)

    if not chronological:
        # permutation happens by some random seeds
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
    else:
        # no permutation happens
        perm = df.index

    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train.reset_index(drop=True).copy(deep=True), validate.reset_index(drop=True).copy(deep=True), test.reset_index(drop=True).copy(deep=True)


class StandardScaler_np:
    """
    An alternative object to replace sklearn's StandardScaler
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.restriction = False

    def transform(self, val):
        return (val - self.mean) / self.std

    def inverse_transform(self, val):
        return val * self.std + self.mean


class NNDataset(Dataset):
    """
    Generate dataset for training environment NN
    """

    def __init__(self, feature, label, device=torch.device("cpu")):
        self.feature = feature.to_numpy()
        self.label = label.to_numpy()
        self.count = feature.shape[0]
        self.device = device

    def __getitem__(self, index):
        feature = torch.from_numpy(self.feature[index]).float().to(self.device)
        label = torch.from_numpy(self.label[index]).float().to(self.device)
        return feature, label

    def __len__(self):
        return self.count


def r2_loss(output, target, target_mean='infer'):
    if target_mean == "infer":
        target_mean = np.mean(target)
    ss_tot = np.sum((target - target_mean) ** 2)
    ss_res = np.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def get_latest_identifier(directory, contains):
    files = os.listdir(directory)
    files = [_ for _ in files if contains in _]
    files = [_ for _ in files if str(holdings_char_flag) in _]
    files = [int(_.split(".")[0].split("_")[-1]) for _ in files]
    return str(max(files))


def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)


def newey_west_adj(series):
    L = int(np.power(len(series), 1 / 3))
    T = len(series)
    mean = np.mean(series)
    demean_series = [_ - mean for _ in series]

    ele_raw = 0.0
    for t in range(1, T + 1):
        ele_raw += demean_series[t - 1] * demean_series[t - 1]

    ele = 0.0
    for l in range(1, L + 1):
        for t in range(l + 1, T + 1):
            wl = 1 - l / (1 + L)
            ele += wl * demean_series[t - 1] * demean_series[t - l - 1]  * 2
    return np.sqrt(1 / T * (ele_raw + ele)), np.sqrt(1 / T * ele_raw)