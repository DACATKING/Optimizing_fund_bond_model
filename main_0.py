"""
Project: Bond Fund Mutual Fund Manager Skills

Step 0: Data Panel Creation
* description:

* inputs:

* variables:

* outputs:


Step 1: Neural Network Training
* description:

* inputs:

* variables:

* outputs:


Step 2: Sensitivity Analysis

"""

from settings import *
from Functions.Utils import log_message, get_latest_identifier, r2_loss, newey_west_adj
from Functions.NNModel import NeuralNetHelper
from Scripts.NNTraining import attach_training_dev
import torch

##################
# DEBUGGING ONLY #
##################
cur = 1
##################
# DEBUGGING ONLY #
##################


if cur <= 0:
    """
    Step 0: 
    """
    pass

if cur <= 1:
    """
    Step 1: train neural network model
    
    Sample: first divide the sample into three equal pieces, two training + dev, one test. Training is 3/4 and dev is
        1/4. 
    Notes: 
        1. ReLU as activation function. 
        2. Each layer has a dropout layer. 
        3. With L1 and L2 regularization in training. Set to be 0 to shut down.
        4. Training with full-batch mode.
        5. Optimizers: Adam
        6. Loss: MSE of future abnormal return
        7. Training epoch: 512
    """

    # define a blank object
    nn_helper = NeuralNetHelper(intermediate_dims=parameters['intermediate_dims'],
                                dropout=parameters['dropout'],
                                learning_rate=parameters['learning_rate'],
                                optimizer=parameters['optimizer'],
                                l1_regularization=parameters['l1_regularization'],
                                l2_regularization=parameters['l2_regularization'])
    identifier = int(time.time())
    sensitivity_collector = {}

    start_time = time.time()

    for ind in range(3):
        # load training dev set
        log_message('Main - load/create training and validation set with test-set id = {}...'.format(ind))
        nn_helper = attach_training_dev(nn_helper, ind, corr_out=ind == 0)  # TODO: bug here, train_x and train_y matrics are not what we expect, missing indicator for holdings-based characteristics

        # define network and training structure
        log_message('Main - define network and training structure...')
        nn_helper.define_network_structure(random_seed_=None)
        # nn_helper.define_network_holdings_structure()

        # Apply torch.compile to optimize the network
#        log_message("Main - compiling the network for optimized training...")
#        nn_helper.network = torch.compile(nn_helper.network)

        # training
        log_message('Main - start training the model...')
        round_start_time = time.time()
        nn_helper.train_minibatch(epochs=parameters['epoch'])
#        log_message('Main - training the model done!')
        round_time = time.time() - round_start_time
        log_message(f"Round {ind} completed in {round_time:.2f} seconds.")

        # save the model
        with lzma.open(network_directory + "nn_helper_{}_{}_{}.xz".format(ind, holdings_char_flag, identifier), 'wb') as f:
            pickle.dump(nn_helper.network, f)

        # sensitivity analysis
        collector_inner = nn_helper.sensitivity()
        sensitivity_collector[ind] = collector_inner
    with open(network_directory + "sensitivity_res_{}_{}.pkl".format(holdings_char_flag, identifier), 'wb') as f:
        pickle.dump(sensitivity_collector, f)

    end_time = time.time()
    elapsed_time = end_time - start_time

    log_message(f"Main: Total training time for the model was {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")

    # assert False

if cur == 2:
    """
    Step 2: Sensitivity Analysis
    """
    if cur == 2:  # just run the sensitivity analysis, load the latest one, otherwise use the focal identifier
        identifier = get_latest_identifier(network_directory, 'nn_helper')

    # with open(network_directory + "sensitivity_res_{}.pkl".format(identifier), 'rb') as f:
    #     sens = pickle.load(f)
    sens = {}
    for i in range(3):
        nn_helper = NeuralNetHelper(intermediate_dims=parameters['intermediate_dims'],
                                    dropout=parameters['dropout'],
                                    learning_rate=parameters['learning_rate'],
                                    optimizer=parameters['optimizer'],
                                    l1_regularization=parameters['l1_regularization'],
                                    l2_regularization=parameters['l2_regularization'])
        nn_helper = attach_training_dev(nn_helper, i, corr_out=i == 0)

        with lzma.open(network_directory + "nn_helper_{}_{}_{}.xz".format(i, holdings_char_flag, identifier), 'rb') as f:
            nn_helper.network = pickle.load(f)

        nn_helper.define_network_structure(init=False)
        # nn_helper.define_network_holdings_structure(init=False)

        sens[i] = nn_helper.sensitivity(delta=1e-5, seeds=range(20))

    collector = {}
    for key in sens[0]:
        collector[key] = [
            np.sqrt(sum([sens[_][key][0] for _ in sens.keys()]) / sum([sens[_][key][1][0] for _ in sens.keys()]))]

    df_sens = pd.DataFrame(collector)
    df_sens.T.to_excel(intermediate_directory + "sensitivity_{}.xlsx".format(identifier))

if cur <= 3:
    """
    Step 3: Portfolio construction and performance
    """
    log_message('Main - Portfolio Construction...')
    if cur == 3:  # just run the portfolio construction, load the latest one, otherwise use the focal identifier
        identifier = get_latest_identifier(network_directory, 'nn_helper')
    df_collect = pd.DataFrame()
    log_message("Main - load models...")
    for i in range(3):
        nn_helper = NeuralNetHelper(intermediate_dims=parameters['intermediate_dims'],
                                    dropout=parameters['dropout'],
                                    learning_rate=parameters['learning_rate'],
                                    optimizer=parameters['optimizer'],
                                    l1_regularization=parameters['l1_regularization'],
                                    l2_regularization=parameters['l2_regularization'])
        nn_helper = attach_training_dev(nn_helper, i, corr_out=i == 0)

        with lzma.open(network_directory + "nn_helper_{}_{}_{}.xz".format(i, holdings_char_flag, identifier), 'rb') as f:
            nn_helper.network = pickle.load(f)
        nn_helper.define_network_structure(init=False)

        df_test = nn_helper.get_test_set_prediction(seeds=range(20))
        df_collect = pd.concat([df_collect, df_test], axis=0)
        print("out of sample R^2 for the subsample: {}".format(r2_loss(df_test['future_ab_monthly_return_pred'], df_test['future_ab_monthly_return'],
            0.0)))

    print("out of sample R^2 for the full sample: {}".format(
        r2_loss(df_collect['future_ab_monthly_return_pred'], df_collect['future_ab_monthly_return'],
                0.0)))

    df_collect.reset_index(drop=True, inplace=True)
    df_collect['cut'] = df_collect.groupby(['Month'], group_keys=False)['future_ab_monthly_return_pred'].apply(lambda x: pd.qcut(x, 10, labels=False))
    df_collect['future_ab_monthly_return'] = np.log(1 + (df_collect['future_ab_monthly_return']))

    # 1: equally weighted
    df_collect_eq_short = df_collect[df_collect['cut'] == 0].groupby('Month')['future_ab_monthly_return'].mean().reset_index()
    df_collect_eq_long = df_collect[df_collect['cut'] == 9].groupby('Month')['future_ab_monthly_return'].mean().reset_index()

    df_collect_eq = pd.merge(df_collect_eq_long.rename(columns={'future_ab_monthly_return': 'long'}), df_collect_eq_short.rename(columns={'future_ab_monthly_return': 'short'}), on='Month', how='inner')
    df_collect_eq['ret'] = df_collect_eq['long'] - df_collect_eq['short']

    df_collect_eq.sort_values('Month').set_index("Month")['ret'].cumsum().plot()
    if terminal_run != 'pycharm':
        plt.show()

    ret_list = df_collect_eq.sort_values('Month').set_index("Month")['ret'].to_list()
    with open(intermediate_directory + "decileport_rets_{}.pkl".format(identifier), 'wb') as f:
        pickle.dump((df_collect, df_collect_eq.sort_values('Month').set_index("Month")['ret']), f)

    log_message('Main: average return is {:.2f}%'.format(100 * np.mean(ret_list)))
    log_message('Main: average std is {:.2f}%'.format(100 * np.std(ret_list)))
    log_message('Main: Sharpe is {:.2f}%'.format(100 * np.mean(ret_list) / np.std(ret_list)))

    log_message('Main: Newey-West adjusted std is {:.2f}%'.format(100 * newey_west_adj(ret_list)[0]))
    log_message('Main: Adjusted Sharpe is {:.2f}%'.format(100 * np.mean(ret_list) / newey_west_adj(ret_list)[0]))

    for i in range(10):
        df_temp = df_collect[df_collect['cut'] == i].groupby('Month')['future_ab_monthly_return'].mean().sort_index().cumsum().plot(label='decile {}'.format(i + 1))
    plt.legend()
    if terminal_run != 'pycharm':
        plt.show()

    # 2: non-equally weighted
    pass

    print("XXXX")

