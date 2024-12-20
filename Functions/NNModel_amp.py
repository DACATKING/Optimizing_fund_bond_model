from settings import *
from Functions.Utils import log_message, r2_loss, NNDataset, init_all


class NeuralNetHelper:
    def __init__(self, intermediate_dims=(20, 40, 20), dropout=0.9, learning_rate=0.001, optimizer='adam',
                 l1_regularization=0.0, l2_regularization=0.0, learning_rate_holdings=0.001,
                 intermediate_dims_holdings=(20, 10)):

        # define attributes as None
        self.network = None
        self.optimizer = None
        self.loss_func = nn.MSELoss()
        self.train_x_full = None
        self.train_x = None
        self.dev_x_full = None
        self.dev_x = None
        self.train_y_full = None
        self.train_y = None
        self.dev_y_full = None
        self.dev_y = None
        self.test_x_full = None
        self.test_x = None
        self.test_y_full = None
        self.test_y = None

        self.standardizer = None

        # define attributes from inputs
        self.intermediate_dims = intermediate_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_def = optimizer
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization

    def define_network_structure(self, init=True, random_seed_=None):
        if init:
            log_message("NN Helper: initialize the network...")
            self.network = NeuralNet(self.train_x.shape[1], self.intermediate_dims, self.dropout)

            if not (random_seed_ is None):
                log_message("NN Helpder: define network initialization manual seed = {}...".format(random_seed_))
                np.random.seed(random_seed_)
                torch.manual_seed(random_seed_)

            init_all(self.network, nn.init.uniform_, -7.5e-2, 7.5e-2)

        if self.optimizer_def == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        elif self.optimizer_def == 'adadelta':
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.learning_rate, rho=0.95, eps=1e-8)
        elif self.optimizer_def == 'momentum':
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            log_message("the optimizer {} is not supported!".format(self.optimizer_def))
            assert False

    def train_minibatch(self, epochs=parameters['epoch'], device="cuda", random_seed_=None):
        log_message("NN Helper: Train neural network...")

        if not (random_seed_ is None):
            log_message("NN Helpder: training set manual seed = {}...".format(random_seed_))
            np.random.seed(random_seed_)
            torch.manual_seed(random_seed_)

        # create data set object
        nn_dataset_obj = NNDataset(self.train_x, self.train_y, torch.device(device))
        self.network.to(torch.device(device))

        scaler = torch.cuda.amp.GradScaler()

        # container
        collector = []
        collector_dev = []
        collector_test = []
        collector_rsquared = []
        collector_dev_rsquared = []
        collector_test_rsquared = []

        nn_dataset_loader = torch.utils.data.DataLoader(dataset=nn_dataset_obj, batch_size=nn_batch_size, shuffle=True)
        self.learning_rate = parameters['learning_rate']
        for epoch in range(epochs):
#            log_message('NN Model - =========================')
#            log_message('NN Model - Epoch {}'.format(epoch))

            # if epoch == [64, 128, 192, 256], haircut learning rate
            if epoch > 31 and epoch % 32 == 0:
                learning_rate_temp = self.learning_rate / pow(2, (epoch // 32))
                self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate_temp)
#                log_message('NN Model - Decreasing learning rate, now: {}'.format(learning_rate_temp))

            # record the full loss (running)
            full_loss = 0.0
            torch.cuda.empty_cache()

            # 1. training mode
            self.network.train()

            counter = 0
            for inputs, labels in tqdm(nn_dataset_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                # clear the grad
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    # feedforward first
                    outputs = self.network(inputs)
                    # then calculate the loss
                    loss = self.loss_func(outputs, labels)

                    all_params = torch.cat([x.view(-1) for x in self.network.parameters()])
                    l1_regularization = self.l1_regularization * torch.norm(all_params, 1)
                    l2_regularization = self.l2_regularization * torch.norm(all_params, 2)
                    loss += l1_regularization + l2_regularization

                # next backward propagate
                scaler.scale(loss).backward()

                # optimize parameters
                scaler.step(self.optimizer)
                scaler.update()
                # add the loss to running loss collector
                full_loss += loss.item()
                counter += 1

#            log_message("{} training loss average compared to benchmark: {}".format(epoch + 1, full_loss / counter))

            # evaluation mode
            self.network.eval()

            # 2. evaluate training set
            outputs = self.network(torch.from_numpy(self.train_x.to_numpy()).float().to(device))

            # then calculate the loss
            loss = self.loss_func(outputs, torch.from_numpy(self.train_y.to_numpy()).float().to(device))
            collector.append(loss.detach().numpy())
            collector_rsquared.append(r2_loss(outputs.detach().numpy(), self.train_y.to_numpy(), target_mean=0.0))

            # 3. evaluate dev set
            # feedforward first
            outputs = self.network(torch.from_numpy(self.dev_x.to_numpy()).float().to(device))

            # then calculate the loss
            loss = self.loss_func(outputs, torch.from_numpy(self.dev_y.to_numpy()).float().to(device))
            collector_dev.append(loss.detach().numpy())
            collector_dev_rsquared.append(r2_loss(outputs.detach().numpy(), self.dev_y.to_numpy(), target_mean=0.0))

            # 4. evaluate test set
            # feedforward first
            outputs = self.network(torch.from_numpy(self.test_x.to_numpy()).float().to(device))

            epoch_time = time.time() - start_time  # Calculate elapsed time
            log_message('NN Model - Epoch {} completed in {:.2f} seconds.'.format(epoch + 1, epoch_time))
            log_message('NN Model - training R^2: {:.4f}, validation R^2: {:.4f}, test R^2: {:.4f}'.format(
                collector_rsquared[-1], collector_dev_rsquared[-1], collector_test_rsquared[-1]))

            # then calculate the loss
            loss = self.loss_func(outputs, torch.from_numpy(self.test_y.to_numpy()).float().to(device))
            collector_test.append(loss.detach().numpy())
            collector_test_rsquared.append(r2_loss(outputs.detach().numpy(), self.test_y.to_numpy(), target_mean=0.0))

#            log_message('NN Model - training R^2: {:.4f}, validation R^2: {:.4f}, test R^2: {:.4f}'.format(
#                collector_rsquared[-1], collector_dev_rsquared[-1], collector_test_rsquared[-1]))

        # get back the correct optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        plt.plot(collector_rsquared[-100:], label='Training')
        plt.plot(collector_dev_rsquared[-100:], label='Validation')
        plt.plot(collector_test_rsquared[-100:], label='Test')
        plt.legend()
        if terminal_run != 'pycharm':
            plt.show()

        return (collector,
                collector_rsquared,
                collector_dev,
                collector_dev_rsquared,
                collector_test,
                collector_test_rsquared
                )

    def sensitivity(self, dataset='test', method='ssd', device='cpu', delta=1e-4, seeds=(0,)):
        assert method in ['ssd']

        if dataset == 'test':
            df_x = self.test_x.copy(deep=True)
        else:
            assert False
        collector = {}
        log_message("Sensitivity: starting...")
        for rd_seed in tqdm(seeds):
            self.network.eval()

            torch.manual_seed(rd_seed)
            np.random.seed(rd_seed)

            out_base = self.network(torch.from_numpy(df_x.to_numpy()).float().to(device)).detach().numpy()

            for col in df_x.keys():
                df_x_deviate = df_x.copy(deep=True)
                df_x_deviate[col] += delta

                torch.manual_seed(rd_seed)
                np.random.seed(rd_seed)

                out_ = self.network(torch.from_numpy(df_x_deviate.to_numpy()).float().to(device)).detach().numpy()
                out_delta = out_ - out_base
                # out_delta *= self.standardizer['future_ab_monthly_return'].std
                # out_delta /= (delta * self.standardizer[col].std)

                if col not in collector:
                    collector[col] = ((out_delta ** 2).sum(), out_delta.shape)
                else:
                    collector[col] = (collector[col][0] + (out_delta ** 2).sum(), collector[col][1] + out_delta.shape)

        return collector

    def get_test_set_prediction(self, device='cpu', seeds=(0,)):
        out_base_collect = []
        for rd_seed in tqdm(seeds):
            self.network.eval()

            torch.manual_seed(rd_seed)
            np.random.seed(rd_seed)
            out_base = self.network(torch.from_numpy(self.test_x.to_numpy()).float().to(device)).detach().numpy()

            out_base_collect.append(out_base)

        df_y = self.test_y_full.copy(deep=True)
        df_y['future_ab_monthly_return_pred'] = sum(out_base_collect) / len(seeds)
        return df_y


class NeuralNet(nn.Module):
    """
    Neural network meta model
    """

    def __init__(self, input_dim, intermediate_dims=(20, 40, 20), dropout=0.9):

        super(NeuralNet, self).__init__()
        self.input_dim = input_dim
        self.intermediate_dims = intermediate_dims
        # define the number of hidden layers
        self.hidden_num = len(intermediate_dims) + 1
        self.dropout = dropout

        # define the first hidden layer
        exec("self.hidden_layer1 = nn.Linear({}, {})".format(input_dim, intermediate_dims[0]))

        # define the following hidden layers except for the last layer
        for i in range(len(intermediate_dims) - 1):
            exec(
                "self.hidden_layer{} = nn.Linear({}, {})".format(i + 2, intermediate_dims[i], intermediate_dims[i + 1]))

        # define the last hidden layer
        exec("self.hidden_layer_last = nn.Linear({}, 1)".format(intermediate_dims[-1]))

    def forward(self, x):
        # use loop to determine the next hidden layers

        for i in range(self.hidden_num - 1):
            x = eval("self.hidden_layer{}(x)".format(1 + i))
            x = F.relu(x)
            x = nn.functional.dropout(x, p=self.dropout)

        y = self.hidden_layer_last(x)
        y = torch.tanh(y)

        return y

    def __repr__(self):
        return "NeuralNet(input_dim={}, output_dim={}, intermediate_dims={}, dropout={})".format(
            self.input_dim.__repr__(), self.output_dim.__repr__(),
            self.intermediate_dims.__repr__(), self.dropout.__repr__()
        )
