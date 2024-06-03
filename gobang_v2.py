from typing import List
from copy import deepcopy
import time
import numpy as np
from policy_value_net import PolicyValueNet 

def softmax(x):
    """softmax function"""
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class node:  # 节点
    def __init__(self, state, player, move=None):
        self.state = deepcopy(state)
        self.player=player
        self.move = move

    def get_state(self):
        return deepcopy(self.state)
    
    def get_player(self):
        return self.player
    
    def get_move(self):
        if self.move:
            return self.move
        else:
            raise Exception("No move trace")


class MCT_node(node):  # MCT节点
    def __init__(self, state, player, prior_p, move=None, parent=None):
        super().__init__(state, player)
        self.parent:MCT_node = parent  # 父节点
        if move: self.move = move
        self.child: list[MCT_node] = []  # 子节点
        self.total_value=0
        self.Q = 0
        self.visits = 0
        self.is_terminal= False
        self.P=prior_p # 先验概率
        self.childlist = [] # 所有可能的（act,probs) 对

    def get_value(self, c_puct = 4): # alpha-zero原论文里的取值，但我认为对于五子棋还可以小一些， medium里的一篇教程说4是不错的选择
        self.u = c_puct * np.sqrt(self.parent.visits) / (1 + self.visits) * self.P  #polynomial upper confidence trees (PUCT) alpha-zero论文里的选择策略
        return self.Q + self.u
    
    def update(self,value):
        self.visits+=1
        self.total_value+=value
        self.Q+=(value-self.Q)/self.visits
    
    def backpropagate(self, value):
        if self.parent:
            self.parent.backpropagate(-value)  #父节点先更新 
        self.update(value)
    #节点是双方交错的，value应当重复地设置正负

    def add_child(self, child): #添加孩子节点
        self.child.append(child)
    
    def is_fully_expanded(self):
        return len(self.childlist) == len(self.child) and len(self.child) > 0


class Game:
    def __init__(self,cell_num=15):
        self.cell_num = cell_num
        self.current_player = 1  # 当前玩家，初始为黑
        self.ingame = False
        self.AI=False
        self.mode=[False]*3
        self.firsthand=1 #黑先手
        self.board = np.array([[0 for _ in range(self.cell_num)] for _ in range(self.cell_num)])  # 棋盘状态，0 表示空，1 表示黑子，2 表示白子
        self.path = [[0, 1], [1, 0], [1, 1], [1, -1]]
        self.past=[]
        self.policy=self.log_policy(self.cell_num)
        self.master=Master(self.cell_num, self.policy)

    def switch_mode(self,AI,mode,firsthand):
        self.AI=AI
        self.mode=mode
        self.firsthand=firsthand
    
    def log_policy(self,cell_num):
        """log model file"""
        model_file = f'../model/current_policy{cell_num}.model' #v2 model # model 路径
        best_policy = PolicyValueNet(self.cell_num, model_file) #policy_param, False) #temp
        return best_policy.policy_value_fn

    def choose_policy(self):
        if self.mode[0]:
            self.master=Master(self.cell_num,self.policy)
        elif self.mode[1]:
            self.master=alphabetaplayer(self.cell_num,self.policy)
        elif self.mode[2]:
            self.master=MCTSplayer(self.cell_num, policy_value_fn=self.policy)

    def start_game(self):
        self.ingame = True
        self.round = 0
        if self.AI and self.firsthand == 2:
            self.put_piece(self.cell_num//2,self.cell_num//2) 

    def Pass(self):
        self.current_player = 3 - self.current_player
        if self.ingame:
            self.get_move()

    def Next(self):
        if self.ingame:
            self.get_move()


    def clear(self):
        self.board = np.array([[0 for _ in range(self.cell_num)] for _ in range(self.cell_num)])
        self.master.clear()
        self.current_player = 1  
        self.mode = [False] * 3
        self.AI = False
        self.ingame = False
        self.past = []


    def regret(self):
        self.ingame = True
        self.past.pop()
        last_node: node = self.past[-2]
        self.board = last_node.get_state()
        self.current_player = last_node.get_player()
        self.past.pop()


    def set_cell_num(self, num):
        self.cell_num = num
        self.board = np.array([[0 for _ in range(self.cell_num)] for _ in range(self.cell_num)])
        self.policy=self.log_policy(self.cell_num)
        self.cell_size = 720 // num


    def put_piece(self, r, c):
        self.board[r, c] = self.current_player
        self.past.append(node(self.board,3- self.current_player,(r,c)))
        if self.success_evaluation(r, c, self.current_player, self.board, self.cell_num):
            self.ingame=False
            return True
        else:
            self.current_player = 3 - self.current_player
        return False

    def get_move(self,r=0,c=0):
        win_rate= -1
        if self.AI:
            r,c, win_rate= self.master.place_piece(self.past[-1])  # AI落子
            self.board[r,c] = self.current_player
            self.past.append(node(self.board, 3 - self.current_player, (r,c)))
            if self.success_evaluation(r, c, self.current_player, self.board, self.cell_num):
                self.ingame = False
                return r,c, win_rate, True
            self.current_player = 3 - self.current_player
        return r,c, win_rate, False


    def test_play(self,player1:'MCTSplayer',player2:'MCTSplayer'):  #模型检验
        """test the performance of the current model"""
        print('start!')
        cell_num=self.cell_num
        state=np.array([[0 for _ in range(self.cell_num)] for _ in range(self.cell_num)]) 
        cur_player=1
        target_x,target_y=cell_num//2, cell_num//2

        state[target_x, target_y] = cur_player
        print(state)

        while not Game.success_evaluation(target_x,target_y,cur_player,state,cell_num):
            cur_player= 3 - cur_player  
            if cur_player==1:
                target_x, target_y, _ = player1.place_piece(node(state,cur_player,(target_x,target_y)))
            else:
                target_x, target_y, _ = player2.place_piece(node(state,cur_player,(target_x,target_y)))

            state[target_x, target_y] = cur_player
            print(state)
            if np.count_nonzero(state) == self.cell_num**2 - 1:
                return  0  #平局    
        if cur_player == 1:
            print(f"winner:testplayer")
        elif cur_player == 2:
            print('winner: trained model')
        else:
            print('tie')
        return cur_player
    

    def self_play(self, player:'MCTSplayer'):  #自我对弈
        """self_play and label the data (s pi z)"""
        cell_num=self.cell_num
        state = np.array([[0 for _ in range(self.cell_num)] for _ in range(self.cell_num)]) 
        cur_player=1
        move, states, mcts_probs, current_players = [], [], [], [] # 数据标注：落子点，棋盘，落子概率，当前玩家
        #第一步
        # target_x,target_y = cell_num//2,cell_num//2
        center_region_size = cell_num // 5  # 中心区域的大小，可以调整
        center = cell_num // 2
        target_x = np.random.randint(center - center_region_size, center + center_region_size)
        target_y = np.random.randint(center - center_region_size, center + center_region_size)
        state[target_x, target_y] = cur_player
        move.append((target_x,target_y))

        while not Game.success_evaluation(target_x,target_y,cur_player,state,cell_num):
            cur_player= 3 - cur_player  
            current_players.append(cur_player)
            states.append(player.feature(state, cur_player, move[-1]))

            target_x, target_y, act_probs = player.place_piece(node(state,cur_player, move[-1]), True)
            state[target_x, target_y] = cur_player
            print(state) # 监视自我对弈进程

            move.append((target_x, target_y))
            mcts_probs.append(act_probs)

            if np.count_nonzero(state) >= self.cell_num**2 - 1 :
                cur_player = 0
                break

        if cur_player == 1:
            print('winner:black')
        elif cur_player == 2:
            print('winner:white')
        else:
            print('tie')
        winners_z = np.zeros(len(move)-1)
        if cur_player: # 非平局
            # winner from the perspective of the current player of each state
            winners_z[np.array(current_players) == cur_player] =  1.0
            winners_z[np.array(current_players) != cur_player] = -1.0
        print(winners_z)
        return zip(states, mcts_probs, winners_z)  #(state,probs,1/-1)


    @staticmethod
    def in_board(x, y, cell_num):
        return x>=0 and y>=0 and x<cell_num and y<cell_num
    
    @staticmethod
    def get_len(r: int, c: int, player: int, path: list, state, cell_num=15):
        len = 1
        dr, dc = path
        for l in range(1, 6):
            nr , nc = r + l * dr, c + l * dc
            if not Game.in_board(nr, nc , cell_num) or state[nr, nc] != player:
                break
            len += 1
        for l in range(1, 6):
            nr , nc = r - l * dr, c - l * dc
            if not Game.in_board(nr, nc , cell_num) or state[nr, nc] != player:
                break
            len += 1
        return len
    
    @staticmethod
    def success_evaluation(i: int, j: int, player: int, state: List[list] = None, cell_num=15):
        path = ((0, 1), (1, 0), (1, 1), (1, -1))
        for x in path:
            cur_len = Game.get_len(i, j, player, x, state, cell_num)
            if cur_len >= 5:
                return True
        return False

    @staticmethod
    def is_valid(r: int, c: int, state, cell_num=15) -> bool:
        return Game.in_board(r, c, cell_num) and state[r, c] == 0


class Master:
    def __init__(self, cell_num: int=8, policy_value_fn=None):
        self.cell_num = cell_num
        self.path = [[0, 1], [1, 0], [1, 1], [1, -1]]
        self.policy = policy_value_fn
    
    def set_cell_num(self,cell_num):
        self.cell_num=cell_num
       

    def clear(self):
        self.weight = np.array([[[self.cell_num // 2 - 0.5 * max((abs(self.cell_num // 2 - i), abs(self.cell_num // 2 - j))) + 1 for i in range(self.cell_num)] 
                                 for j in range(self.cell_num)] for _ in range(2)])  # 基础分 #4.19 黑白分权，0黑1白


    def feature(self, board:np.ndarray, player, last_move:tuple=None): # 将棋局分解成特征平面
        """turn a state into three feature planes"""
        cell_num=len(board[0])
        square_state = np.zeros((3, cell_num, cell_num))
        square_state[0] = np.where(board==player, 1, 0) # 我方落子(对于当前的棋盘即将落子的一方)
        square_state[1] = np.where(board==3-player, 1, 0) # 对手落子
        if last_move:
            square_state[2][last_move]=1  # focus，最后一处落子
        return square_state


    def greedy(self, node) -> tuple:
        try:
            goal = self.extend_frontier(node, 1)[0][0]
        except:
            raise Exception('unknown error')
        return goal


    def extend_frontier(self, node:node, num=None):
        # 计算有效的可选位置数
        len_available = len(node.state[0])**2 - np.count_nonzero(node.state)
        if len_available == 0:
            return [[(-1, -1), 0]]
        if num:
            num = min(len_available, num)
        else:
            num = len_available
        act_probs:list = self.policy(node)[0]
        act_probs.sort(key=lambda x: x[1].mean(), reverse=True)
        return act_probs[:num]
        # 根据概率降序排序
        '''
        top_k_coordinates = []
        previous_prob = None
        threshold = 2  #阈值
        for i, (coord, prob) in enumerate(act_probs):
            if previous_prob is None:
                top_k_coordinates.append((coord, prob))
            else:
                if previous_prob.mean() > threshold * prob.mean():
                    break 
                top_k_coordinates.append((coord, prob))
            previous_prob = prob
            if i >= num - 1:
                break
        return top_k_coordinates
        '''


    def place_piece(self, node:node):
        target_x, target_y = self.cell_num // 2, self.cell_num // 2
        target_x, target_y = self.greedy(node)
        if Game.is_valid(target_x, target_y, node.state, self.cell_num):
            return target_x, target_y, -1
        else:
            raise Exception(f'no valid move:({target_x},{target_y})')  



class alphabetaplayer(Master):

    def __init__(self, cell_num: int, policy_value_fn, search_depth: int=2, search_num: int=2):
        super().__init__(cell_num, policy_value_fn)
        self.search_depth = search_depth
        self.option = search_num
    
    def switch_mode(self,depth,num):
        self.search_depth=depth
        self.option=num


    def place_piece(self, node:node):
        target_x, target_y = self.cell_num // 2, self.cell_num // 2
        target_x, target_y = self.alpha_beta_search(node)
        if Game.is_valid(target_x, target_y, node.state, self.cell_num):
            return target_x, target_y, -1
        else:
            raise Exception(f'no valid move:({target_x},{target_y})') 
    

    def alpha_beta_search(self, node:node):
        target_x, target_y = 0, 0
        alpha, beta = float('-inf'), float('inf')
        if node.player == 1:
            path = self.max_value(node, alpha, beta, 1)[1]
        else:
            path = self.min_value(node, alpha, beta, 1)[1]
        if len(path) > 0:
            target_x, target_y = path[0]
        if Game.is_valid(target_x, target_y, node.state, self.cell_num):
            return target_x, target_y
        else:
            raise Exception("Invalid position")


    def min_value(self, node:node, alpha: float, beta: float, depth: int):  # 白棋
        if depth > self.search_depth:
            return self.policy(node)[1], []
        else:
            max_num = max(max(6, self.option) // (depth + 1), 2)
            frontier = self.extend_frontier(node, max_num)
            v, move, path = float('inf'), (-1, -1), []
            for x, _ in frontier: #act probs
                e_x, e_y = x
                node.state[e_x, e_y] = 2
                if Game.success_evaluation(e_x, e_y, 2, node.state, self.cell_num):
                    utility = -1 # 白棋获胜
                    node.state[e_x, e_y] = 0
                    return utility, [x]  # utility
                node.player = 1
                v2, temppath = self.max_value(node, alpha, beta, depth + 1)
                if v2 < v:
                    v = v2
                    move = x
                    path = [move] + temppath
                beta = min(beta, v)
                node.player=2
                node.state[e_x, e_y] = 0
                if v <= alpha:
                    return v, path
            return v, path


    def max_value(self, node:node, alpha: float, beta: float, depth: int):  # 黑棋
        if depth > self.search_depth:
            return self.policy(node)[1], []
        else:
            max_num = max(max(6, self.option) // (depth + 1), 2)
            frontier = self.extend_frontier(node, max_num)
            v, move, path = float('-inf'), (-1, -1), []
            for x, _ in frontier:
                e_x, e_y = x
                node.state[e_x,e_y] = 1
                if Game.success_evaluation(e_x, e_y, 1, node.state, self.cell_num):
                    utility = 1
                    node.state[e_x, e_y] = 0
                    return utility, [x]  
                node.player=2
                v2, temppath = self.min_value(node, alpha, beta, depth + 1)
                if v2 > v:
                    v = v2
                    move = x
                    path = [move] + temppath
                alpha = max(alpha, v)
                node.player=1
                node.state[e_x, e_y] = 0
                if v >= beta:
                    return v, path
            return v, path
        

class MCTSplayer(Master):

    def __init__(self, cell_num, n_rollout=100, is_selfplay=False, policy_value_fn=None):
        super().__init__(cell_num, policy_value_fn)
        self.n_rollout=n_rollout
        self.search_depth = 12
        self.is_selfplay = is_selfplay
        self.root=MCT_node(np.zeros((self.cell_num, self.cell_num)), player=1, prior_p=1) #当前根节点，用于追踪和重用树
    
    def switch_mode(self,depth,num,n_rollout): #2.0不接受num参数
        self.search_depth = depth
        self.n_rollout = n_rollout


    def place_piece(self, node:node, return_prob=False):
        target_x, target_y = self.cell_num // 2, self.cell_num // 2
        if return_prob:
            target_x,target_y, mcts_probs = self.MCTS_train(node)
            if Game.is_valid(target_x, target_y, node.state, self.cell_num):
                return target_x, target_y, mcts_probs
            else:
                raise Exception(f'not valid move:({target_x},{target_y})')
        else:
            target_x, target_y, winrate = self.MCTS_pure(node)
            if Game.is_valid(target_x, target_y, node.state, self.cell_num):
                return target_x, target_y, winrate
            else:
                raise Exception(f'not valid move:({target_x},{target_y})') 


    def MCTS_train(self, node:node):
        """MCTS process for training, return move and prob matrix"""
        root = MCT_node(node.state, node.player, prior_p=1, move=node.move)
        # mcts
        time_limit = 20  # s-时间上限
        start_time = time.time()
        sum = 0
        while sum < self.n_rollout:
            self.execute(root) 
            sum += 1
            if time.time() - start_time > time_limit:
                break

        # 计算动作及概率
        act_visits = [(x, x.visits) for x in root.child]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/(1e-3) * np.log(np.array(visits) + 1e-10))

        if self.is_selfplay:
                # \epsilon-greedy and Dirichlet Noise (for self-play training)
                act_probs=0.75*act_probs + 0.25*np.random.dirichlet(1*np.ones(len(act_probs)))
        
        choice:MCT_node = np.random.choice(np.array(acts), p = act_probs) # 约等于选择探索次数最多的节点
        target_x,target_y=choice.get_move()

        mcts_probs = np.zeros(self.cell_num*self.cell_num).reshape(self.cell_num, self.cell_num)
        move_probs=list(zip(acts, act_probs))
        for act, prob in move_probs:
            x, y = act.get_move()
            mcts_probs[x,y] = prob # 计算探索所有动作的概率

        return target_x, target_y, mcts_probs


        
    def MCTS_pure(self, node:node):
        root=MCT_node(node.state, node.player, prior_p=1, move=node.move)
    
        time_limit = 30  # s时间上限
        start_time = time.time()
        sum = 0
        while sum < self.n_rollout:
            self.execute(root) 
            sum += 1
            if time.time() - start_time > time_limit:
                break
        
        act_visits = [(x, x.visits) for x in root.child]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/(1e-3) * np.log(np.array(visits) + 1e-10)) # sum = 1

        choice:MCT_node = np.random.choice(np.array(acts), p=act_probs)
        target_x,target_y = choice.get_move()
        winrate = 0.5 + 0.5 * choice.Q
        return target_x, target_y, winrate
    

    def execute(self, node: MCT_node):
        while node.visits and not node.is_terminal:  # 逐层选择，直到未完全拓展的节点，返回它的一个新叶子
            node = self.select(node)
        if not node.is_terminal:
            result = self.rollout(node)
            node.backpropagate(result)
        else:
            node.backpropagate(1)
    
    def expand(self,node:MCT_node): #异步拓展， 每经过一次增加一个子节点，直到完全拓展
        if not node.childlist: #第一次探索，获取所有可能的孩子节点
            node.childlist = self.extend_frontier(node)
        action, prob = node.childlist[len(node.child)] 
        e_x,e_y = action
        node.state[e_x,e_y] = node.player
        if type(prob) == np.ndarray: prob = prob.mean()
        node.add_child(MCT_node(node.state, 3 - node.player, prob, action , node))
        if Game.success_evaluation(e_x, e_y, node.player, node.state, self.cell_num): 
            # 终止游戏：如果存在终态，则必然达到终态，不必继续扩展和探索这个节点 
            node.child[-1].is_terminal=True
            node.child[-1].backpropagate(1)
        node.state[e_x,e_y] = 0


    def rollout(self, node:MCT_node):
        depth = 0
        roll = deepcopy(node)
        originplayer = roll.player
        while True:
            e_x,e_y = self.greedy(roll)
            roll.state[e_x,e_y] = roll.player
            if Game.success_evaluation(e_x, e_y, roll.player, roll.state, self.cell_num):
                break
            roll.player = 3 - roll.player
            depth += 1
            if roll.state.shape[0]**2 - np.count_nonzero(roll.state) < 4 or depth > self.search_depth:
                return self.policy(roll)[1] # ?
        if roll.player == originplayer: 
            return -1
        else:
            return 1
    

    def select(self,node:MCT_node) -> MCT_node: # 由根至叶，选择没有经过rollout的节点
        if node.is_terminal:
            return node
        elif not node.is_fully_expanded(): # 没有完全拓展，增加子节点
            self.expand(node) # 增加一个子节点
            return node.child[-1] # 返回新增添的子节点
        else: #已经探索了全部子节点，按照PUCB方法选择
            return max(node.child, key=lambda x: x.get_value())
