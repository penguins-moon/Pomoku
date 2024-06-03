import random
import numpy as np
from collections import defaultdict, deque
from gobang_v2 import Game, MCTSplayer 
from policy_value_net import PolicyValueNet 
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys

class Train():
    def __init__(self, cell_num=9, init_model=None):
        self.cell_num = cell_num
        self.game = Game(self.cell_num)
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 根据KL散度，动态调整学习率
        self.n_playout = 200 # 训练模拟次数
        self.buffer_size = 12000 #缓存样本容量
        self.batch_size = 512  # 每次训练的batch_size
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 25  # 每次更新策略的迭代次数
        self.kl_targ = 0.02
        self.save_freq = 10 
        self.check_freq = 100  
        self.game_batch_num = 1000 # 一次训练的总批次数
        self.best_win_ratio = 0.0
        self.test_mcts_playout_num = 400 # testplayer模拟次数
        if init_model:
            self.policy_value_net = PolicyValueNet(self.cell_num, model_file=init_model)
        else:
            # 继续训练
            self.policy_value_net = PolicyValueNet(self.cell_num)
        self.pure_net = PolicyValueNet(self.cell_num)
        self.mcts_player = MCTSplayer(self.cell_num, self.n_playout, True, self.policy_value_net.policy_value_fn)


    def get_equi_data(self, play_data): # 旋转对称，扩增数据集
        """rotate and flip, extend the date set"""
        extend_data = []
        for state, mcts_porb, player in play_data:
            for i in [1, 2, 3, 4]:
                #旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(mcts_porb, i)
                extend_data.append((equi_state, equi_mcts_prob, player))
                #水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,equi_mcts_prob,player))
        return extend_data

    
    def collect_selfplay_data(self, n_games=1): #收集自我对弈数据
        for _ in range(n_games):
            play_data = self.game.self_play(self.mcts_player)
            play_data = list(play_data) # 转化为列表
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            self.episode_len = len(play_data)
            print("finish collecting self-play data", len(self.data_buffer))

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = np.array([data[2] for data in mini_batch])
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch, 
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            print('epoches:', i+1, 'loss:', loss, 'entropy:', entropy)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),axis=1))
            if kl > self.kl_targ * 4:  # D_KL偏离太大，提前终止
                break
        #调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        #解释方差
        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / (1e-16 + np.var(np.array(winner_batch))))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / (1e-16 + np.var(np.array(winner_batch))))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy


    def policy_evaluate(self, n_games=5): #测试模型
        """test the model"""
        current_mcts_player = MCTSplayer(self.cell_num, self.n_playout, False, self.policy_value_net.policy_value_fn)
        test_player =  MCTSplayer(self.cell_num, self.test_mcts_playout_num, False, self.pure_net.policy_value_fn) 
        win_cnt = defaultdict(int)
        for _ in range(n_games):
            winner = self.game.test_play(test_player,current_mcts_player)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[2] + 0.5*win_cnt[0]) / n_games # 计算胜率
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.test_mcts_playout_num,
                win_cnt[2], win_cnt[1], win_cnt[0]))
        return win_ratio
    
    def run(self):
        """run the training pipeline"""
        try:
            losses = []
            entropies = []
            sns.set_theme(style="darkgrid")
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 6))
            line1, = ax.plot(losses, label='Loss')
            line2, = ax.plot(entropies, label='Entropy')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Value')
            ax.legend()
            
            start_time = time.time()
            print('start training')
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size) #收集自我对弈数据
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update() #更新策略
                    losses.append(loss)
                    entropies.append(entropy)
                    
                    line1.set_ydata(losses) # 更新曲线
                    line1.set_xdata(range(len(losses)))
                    line2.set_ydata(entropies)
                    line2.set_xdata(range(len(entropies)))
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw_idle()
                    plt.pause(0.5)
                    plt.gcf().canvas.flush_events()
                    
                if (i+1) % self.save_freq == 0:
                    self.policy_value_net.save_model(f'../model/current_policy{self.cell_num}.model')
                    print('model saved')
                    
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    print('evaluating policy...')
                    win_ratio = self.policy_evaluate() # 模拟测试，检验模型性能
                    if win_ratio > self.best_win_ratio: 
                        print("New best policy")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(f'../model/best_policy{self.cell_num}.model')
                        if self.best_win_ratio >= 0.9:
                            self.test_mcts_playout_num *= 2
                            self.best_win_ratio = 0.0
                    else:
                        print('oops')

            elapsed_time = time.time() - start_time
            print(f'Training completed in: {elapsed_time:.2f} seconds')
        except KeyboardInterrupt:
            print('\n\rquit')
        finally:
            plt.ioff()  
            plt.show()


if __name__ == '__main__':
    # os.chdir(sys.path[0])
    cell_num = 9
    training = Train(cell_num, f'../model/current_policy{cell_num}.model')
    training.run()
