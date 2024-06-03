import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, cell_num):
        super(Net, self).__init__()
        self.cell_num=cell_num
        # 通用卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 策略头
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1) # 1*1 filter, 降维
        self.act_fc1 = nn.Linear(4*self.cell_num**2, self.cell_num**2)
        # 价值头
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*self.cell_num**2, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        """forward"""
        #前向传播
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.cell_num**2)
        x_act = F.softmax(self.act_fc1(x_act), dim=1) # shape:[1,64]
        # 价值层， tanh转化到[-1,1]
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.cell_num**2)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val)) # shape [1,1]
        return x_act, x_val

'''
class ResidualBlock(nn.Module):
    """残差块，包含两个卷积层和一个跳过连接"""
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                #nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Net(nn.Module):
    """策略-价值网络模块"""
    def __init__(self, cell_num):
        super(Net, self).__init__()
        self.cell_num = cell_num
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        #self.bn1 = nn.BatchNorm2d(32)
        # 残差块
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.res_block3 = ResidualBlock(128, 128)  
        self.res_block4 = ResidualBlock(128, 128)
        # 策略头
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * cell_num**2, cell_num**2)
        # 价值头
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * cell_num**2, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        x = F.relu(self.conv1(state_input))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        # 策略层
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.cell_num**2)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # 价值层
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.cell_num**2)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val
'''


class PolicyValueNet():
    """policy-value network """
    def __init__(self,cell_num, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.cell_num=cell_num
        self.l2_const = 1e-4  # coef of l2 penalty
        #加载网络
        if self.use_gpu:
            self.policy_value_net = Net(self.cell_num).cuda()
        else:
            self.policy_value_net = Net(self.cell_num)
        #优化器： Adam
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)
        #加载模型
        if model_file:
            self.log_model(model_file)
    
    def log_model(self, model_file):
        """load the model(if any)"""
        try:
            with open(model_file, 'rb') as f:
                try:
                    policy_param = torch.load(f)
                except UnicodeDecodeError:
                    # 如果有编码错误，尝试用 encoding='bytes'
                    f.seek(0)  # 重置文件指针
                    policy_param = torch.load(f, encoding='bytes')
        except FileNotFoundError:
            print(f"Error: The file {model_file} does not exist.")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        self.policy_value_net.load_state_dict(policy_param)
        print("Model loaded successfully")


    def policy_value(self, state_batch):  # 训练用
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda()) # state_batch:list[data[0] in (state, act_porbs, winner)]
            act_probs, value = self.policy_value_net(state_batch)
            return act_probs.data.cpu().numpy(), value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            act_probs,value = self.policy_value_net(state_batch)
            return act_probs.data.numpy(), value.data.numpy()


    def feature(self, board:np.ndarray, player, last_move:tuple=None): # 将当前局面拆解为特征平面
        """turn a state into three feature planes"""
        square_state = np.zeros((3, self.cell_num, self.cell_num))
        square_state[0] = np.where(board==player, 1, 0) #我方落子
        square_state[1] = np.where(board==3-player, 1, 0) #对手落子
        if last_move: square_state[2][last_move] = 1 #focus
        return square_state


    def policy_value_fn(self, node): 
        """
        input: node
        output: (action, probability)，state values
        """
        try:
            board=node.get_state()
            player=node.get_player()
            last_move=node.get_move()
        except:
            raise Exception('not a valid node')
        legal_positions = np.transpose(np.where(board==0)).tolist()  # 空位坐标
        current_state = self.feature(board, player, last_move)
        if self.use_gpu:
            act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = act_probs.data.cpu().numpy()[0,:] 
            value = value.data.cpu().numpy()[0][0]
        else:
            act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = act_probs.data.numpy()[0,:]
            value = value.data.numpy()[0][0]
        act_probs = list(zip(legal_positions, act_probs[legal_positions]))
        return act_probs, value
    

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """execute the training step"""
        #将样本转化为为torch Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # 清除梯度缓存
        self.optimizer.zero_grad()
        # 设置学习率
        set_learning_rate(self.optimizer, lr)

        # forward
        act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        mcts_probs=mcts_probs.reshape(mcts_probs.shape[0], -1)
        policy_loss = - torch.mean(torch.sum(mcts_probs*torch.log(act_probs), 1))
        loss = value_loss + policy_loss
        # backward
        loss.backward()
        self.optimizer.step()
        # 计算平均熵
        entropy = -torch.mean(torch.sum(act_probs* torch.log(act_probs), 1))
        return loss.item(), entropy.item()


    def get_policy_param(self): 
        net_params = self.policy_value_net.state_dict()
        return net_params 


    def save_model(self, model_file):
        net_params = self.get_policy_param()  # 保存模型参数
        torch.save(net_params, model_file)


def set_learning_rate(optimizer, lr): # 设置学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr