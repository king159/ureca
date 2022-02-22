import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import torch.optim as optim
import optuna
from util import topk_acc

seed = 0
torch.manual_seed(seed)
device = torch.device("cuda")


class Manual(nn.Module):
    def __init__(self):
        super(Manual, self).__init__()

        self.fc1_1 = self.fc_block(19, 19)
        self.fc1_2 = self.fc_block(216, 216)
        self.fc1_3 = self.fc_block(128, 128)
        self.fc2 = self.fc_block(363, 256)
        self.fc3 = self.fc_block(256, 7, final_layer=True)
        self.dp = nn.Dropout(0.2)

    def fc_block(self, input_dims, output_dims, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_dims, output_dims),
                nn.ReLU(True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dims, output_dims),
                # nn.Dropout(0.1),
            )

    def forward(self, x):
        x = torch.split(x, [19, 216, 128], dim=1)
        x_1 = self.fc1_1(x[0])
        x_2 = self.fc1_2(x[1])
        x_3 = self.fc1_3(x[2])
        x = torch.cat([x_1, x_2, x_3], dim=1)
        x = self.dp(self.fc2(x))
        x = self.fc3(x)
        return x


class Auto(nn.Module):
    def __init__(self):
        super(Auto, self).__init__()
        self.encoder = nn.Sequential(
            self.fc_block(773, 600),
            self.fc_block(600, 400),
            self.fc_block(400, 200),
            self.fc_block(200, 100),
        )
        self.fc = nn.Sequential(
            self.fc_block(100, 1024),
            self.fc_block(1024, 512),
            self.fc_block(512, 256),
            self.fc_block(256, 7, final_layer=True)
        )

    def fc_block(self, in_dim, out_dim, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(0.1),
                nn.ReLU(True)
            )
        else:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(0.1)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.cnn = nn.Sequential(
            self.cnn_block(12, 96),
            self.cnn_block(96, 256),
            self.cnn_block(256, 384),
            self.cnn_block(384, 256),
        )
        self.fc = nn.Sequential(
            self.fc_block(256, 128, 0.2),
            self.fc_block(128, 7, 0.2, final_layer=True),
        )

    def cnn_block(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(output_channels),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(True),
        )

    def fc_block(self, input_dim, output_dim, dropout=0.0, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Dropout(dropout),
                nn.ReLU(True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        x = self.up(x)
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            length: int
    ) -> None:
        super(InceptionBlock, self).__init__()
        out_channels = int(out_channels / 4)
        self.branch1x1 = self.conv_block(in_channels, out_channels, kernel_size=(1, 1))

        self.branch3x3 = self.conv_block(in_channels, out_channels, kernel_size=(3, 3), padding=1)

        self.branchLx1 = self.conv_block(in_channels, out_channels, kernel_size=(length, 1))
        self.branch1xL = self.conv_block(in_channels, out_channels, kernel_size=(1, length))

        self.branch_pool = self.conv_block(in_channels, out_channels, kernel_size=(1, 1))

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3(x)

        branchLx1 = self.branchLx1(x)
        branch1xL = self.branch1xL(x)
        branchLxL = torch.matmul(branch1xL, branchLx1)

        branch_pool = self.branch_pool(F.avg_pool2d(x, kernel_size=(3, 3), stride=1, padding=1))

        outputs = [branch1x1, branch3x3, branchLxL, branch_pool]
        return outputs

    def conv_block(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            # nn.Dropout2d(0.2)
        )

    def forward(self, x):
        x = self._forward(x)
        x = torch.cat(x, 1)
        return x


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.inception1 = InceptionBlock(12, 64, 8)
        self.inception2 = InceptionBlock(64, 192, 8)
        self.inception3 = InceptionBlock(192, 384, 8)

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc = nn.Sequential(
            self.fc_block(1536, 1024),
            self.fc_block(1024, 512),
            self.fc_block(512, 256),
            self.fc_block(256, 7, final_layer=True)
        )

    def fc_block(self, input_dim, output_dim, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Dropout(0.2),
                nn.ReLU(True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                # nn.Dropout(0.2),
            )

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)

        x = self.avgpool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Trivial(nn.Module):
    def __init__(self):
        super(Trivial, self).__init__()
        self.fc = nn.Sequential(
            self.fc_block(773, 1024, 0.25),
            self.fc_block(1024, 512, 0.25),
            self.fc_block(512, 256, 0.25),
            self.fc_block(256, 7, 0.25, final_layer=True),
        )

    def fc_block(self, input_dim, output_dim, dropout=0.0, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Dropout(dropout),
                nn.ReLU(True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        x = self.fc(x)
        return x


def get_model(types: str):
    # channels = [trial.suggest_int("cnn1", 1, 1024), trial.suggest_int("cnn2", 1, 1024),
    #             trial.suggest_int("cnn3", 1, 1024), trial.suggest_int("cnn4", 1, 2048)]
    # dropout = trial.suggest_float('dropout', 0, 0.5, step=0.05)
    factory = {'manual': Manual(), 'cnn': CNN(), 'auto': Auto(), 'inception': Inception(),
               'trivial': Trivial()}
    net = factory[types]
    if types == 'auto':
        try:
            net.load_state_dict(torch.load('./model/encoder.pt'))
        except RuntimeError:
            pass
    return net


def get_optimizer(trial, model):
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    if optimizer_name == 'Adam':
        adam_lr = trial.suggest_float("adam_lr", 1e-5, 1e-1, log=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    else:
        wd = trial.suggest_float("wd", 1e-4, 1e-2, log=True)
        sgd_lr = trial.suggest_float("sgd_lr", 1e-5, 1e-1, log=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=sgd_lr, momentum=0.9, weight_decay=wd)
    return optimizer


def get_data():
    data_pt = torch.load('./small_data/cnn2_c.pt')
    train_set, dev_set = sklearn.model_selection.train_test_split(data_pt, test_size=0.1, train_size=0.9)

    datasets = {'train': train_set, 'dev': dev_set}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'dev']}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1024, shuffle=True, pin_memory=True)
                   for x in ['train', 'dev']}

    return dataloaders, dataset_sizes


def objective(trial):
    net = get_model('cnn').to(device)
    num_epochs = 60
    lr_decay = trial.suggest_float("lr_decay", 0.95, 0.99, step=0.01)
    optimizer = get_optimizer(trial, net)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    criterion = nn.CrossEntropyLoss()
    dataloaders, dataset_sizes = get_data()

    acc = 0.0
    for epoch in trange(num_epochs):
        net.train()
        for batch in dataloaders['train']:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        net.eval()
        correct = 0
        with torch.no_grad():
            for batch in dataloaders['dev']:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = net(inputs)
                correct += topk_acc(outputs, labels, 1)
        acc = correct / dataset_sizes['dev']
        # pruning
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(),
                                sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100)
    # fig1 = optuna.visualization.plot_slice(study, params=["sgd_lr", "lr_decay"])
    # fig1.show()
    # fig2 = optuna.visualization.plot_slice(study, params=["adam_lr"])
    # fig2.show()

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
