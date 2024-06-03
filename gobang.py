from typing import List
from copy import deepcopy
import time
import numpy as np

class node:  # 节点
    def __init__(self, state, player, weight, move=None):
        self.state = deepcopy(state)
        self.player=player
        self.move = move
        self.weight = deepcopy(weight)

    def get_state(self):
        """return the chessboard state(deepcopy)"""
        return deepcopy(self.state)

    def get_weight(self):
        """return the weight matrix(deepcopy)"""
        return deepcopy(self.weight)
    
    def get_player(self):
        """return the current player"""
        return self.player
    
    def get_move(self):
        """return the last action"""
        if self.move:
            return self.move
        else:
            raise Exception("No move trace")


class MCT_node(node):  # MCT节点
    def __init__(self, state, player, weight, move=None, parent=None, depth=0):
        super().__init__(state, player, weight)
        self.parent:MCT_node = parent  # 父节点
        if move: self.move = move
        self.child: list[MCT_node] = []  # 子节点
        self.Q = 0
        self.visits = 0
        self.depth = depth  # 搜索树中的深度
        self.is_terminal= False # 游戏结束
        self.childlist = []

    def get_value(self, c_puct=1):
        """return the value: Q + U"""
        self.u = c_puct * np.sqrt(np.log(self.parent.visits)/ self.visits)
        # self.u = c_puct * np.sqrt(self.parent.visits) / (1 + self.visits)
        return self.Q + self.u
    
    def update(self,value):
        """update this node"""
        self.visits+=1
        self.Q += (value-self.Q)/self.visits
        # self.Q = self.Q+(value-self.Q)/self.visits = (self.Q*(self.visits-1) + 1*self.value)/self.visits
    
    def backpropagate(self, value):
        """send the rollout result backward"""
        self.update(value)  #对手节点的价值是相反的
        if self.parent:
            self.parent.backpropagate(-value)
    
    def add_child(self, child):
        """add a new child node to the child list"""
        self.child.append(child)

    def is_fully_expanded(self):
        """check if all actions have been appended to the child list"""
        return len(self.childlist) == len(self.child) and len(self.child) > 0


class Game:
    def __init__(self,cell_num=15):
        self.cell_num = cell_num
        self.current_player = 1  # 当前玩家，初始为黑
        self.ingame = False
        self.AI=False
        self.mode=[False]*3
        self.firsthand = 1 #黑先手
        self.board = np.array([[0 for _ in range(self.cell_num)] for _ in range(self.cell_num)])  # 棋盘状态，0 表示空，1 表示黑子，2 表示白子
        self.past=[] 
        self.master=Master()

    def switch_mode(self,AI,mode,firsthand):
        """choose AI/human,firsthand"""
        self.AI=AI
        self.mode=mode
        self.firsthand=firsthand

    def choose_policy(self): 
        """choose AI type"""
        if self.mode[0]:
            self.master=Master(self.cell_num)
        elif self.mode[1]:
            self.master=alphabetaplayer(self.cell_num)
        elif self.mode[2]:
            self.master=MCTSplayer(self.cell_num)

    def start_game(self):
        """start game"""
        self.ingame = True
        self.round = 0
        if self.AI and self.firsthand == 2:
            self.put_piece(self.cell_num//2,self.cell_num//2) 

    def Pass(self):
        """pass the current round"""
        self.current_player = 3 - self.current_player
        if self.ingame:
            self.get_move()

    def Next(self):
        """AI takes the next move"""
        if self.ingame:
            self.get_move()


    def clear(self):
        """reset the game"""
        self.board = np.array([[0 for _ in range(self.cell_num)] for _ in range(self.cell_num)])
        self.master.clear()
        self.current_player = 1  
        self.mode = [False] * 3
        self.AI = False
        self.ingame = False
        self.past = []


    def regret(self): #悔棋
        """regret before it's too late"""
        self.ingame = True
        self.past.pop()
        last_node: node = self.past[-2]
        self.board = last_node.get_state()
        self.master.weight = last_node.get_weight()
        self.current_player = last_node.get_player()
        self.past.pop()


    def set_cell_num(self, num):
        """reset the board size"""
        self.cell_num = num
        self.board = np.array([[0 for _ in range(self.cell_num)] for _ in range(self.cell_num)])
        self.cell_size = 720 // num


    def put_piece(self, r, c):
        """place a piece in the annotated position"""
        self.board[r, c] = self.current_player
        self.master.update_weight(r,c,self.board,self.master.weight) #test
        self.past.append(node(self.board,3- self.current_player,self.master.weight,(r,c)))
        if self.success_evaluation(r, c, self.current_player, self.board, self.cell_num):
            self.ingame=False
            return True
        else:
            self.current_player = 3 - self.current_player
        return False

    def get_move(self,r=0,c=0): # r,c: the last move position
        """get the action, winrate from AI and check end state"""
        win_rate= -1
        if self.AI:
            r,c, win_rate= self.master.place_piece(self.current_player, self.board)  # AI落子
            self.board[r,c] = self.current_player
            self.master.update_weight(r,c,self.board,self.master.weight)
            self.past.append(node(self.board, 3 - self.current_player,self.master.weight,(r,c)))
            if self.success_evaluation(r, c, self.current_player, self.board, self.cell_num):
                self.ingame = False
                return r,c, win_rate, True
            self.current_player = 3 - self.current_player
        return r,c, win_rate, False
    

    @staticmethod
    def in_board(x, y, cell_num):
        """if the coordinate is out of the chessboard"""
        return x>=0 and y>=0 and x<cell_num and y<cell_num
    
    @staticmethod
    def get_len(r: int, c: int, player: int, path: list, state, cell_num=15):
        """return the link length in the direction"""
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
        """end state judgement"""
        path = ((0, 1), (1, 0), (1, 1), (1, -1))
        for x in path:
            cur_len = Game.get_len(i, j, player, x, state, cell_num)
            if cur_len >= 5:
                return True
        return False

    @staticmethod
    def is_valid(r: int, c: int, state, cell_num=15) -> bool:
        """valid move check"""
        return Game.in_board(r, c, cell_num) and state[r, c] == 0
    
    


class Master:
    def __init__(self, cell_num: int=15):
        self.cell_num = cell_num
        self.weight = np.array([[[self.cell_num // 2 - 0.25 * max((abs(self.cell_num // 2 - i), abs(self.cell_num // 2 - j))) + 1 for i in range(self.cell_num)] 
                                 for j in range(self.cell_num)] for _ in range(2)])  # 基础分 #4.19 黑白分权，0黑1白
        self.base = np.array([[self.cell_num//2 - 0.25 * max((abs(self.cell_num // 2 - i), abs(self.cell_num // 2 - j))) + 1 for i in range(self.cell_num)] for j in range(self.cell_num)])  # 控制距离系数

    
    def set_cell_num(self,cell_num):
        """reset the size of weight matrix"""
        self.cell_num=cell_num
        self.weight = np.array([[[self.cell_num // 2 - 0.5 * max((abs(self.cell_num // 2 - i), abs(self.cell_num // 2 - j))) + 1 for i in range(self.cell_num)] 
                                 for j in range(self.cell_num)] for _ in range(2)]) 
        self.base = np.array([[self.cell_num//2 - 0.5 * max((abs(self.cell_num // 2 - i), abs(self.cell_num // 2 - j))) + 1 for i in range(self.cell_num)] for j in range(self.cell_num)])
    

    def clear(self):
        """reset the weight matrix"""
        self.weight = np.array([[[self.cell_num // 2 - 0.5 * max((abs(self.cell_num // 2 - i), abs(self.cell_num // 2 - j))) + 1 for i in range(self.cell_num)] 
                                 for j in range(self.cell_num)] for _ in range(2)])

    def score(self, line: str) -> int:
        """score of a single line based on expert knowledge"""
        lines_dict = {'11111': 1000000, '22222': -1000000, 
                      '011110': 1000, '022220': -1000, 
                      '011100': 150, '022200': -150, 
                      '011010': 135, '022020': -135, 
                      '211110': 180, '122220': -180, 

                      '11011': 180, '22022': -180, 
                      # '11011100': -30, '22022200': 30, 
                      # '11011010': -30, '22022020': 30,
                      '10111': 180,'20222': -180,
                      # '1011100': -30, '2022200': 30, 
                      # '011010111': -30, '022020222': 30, 
                      # '010110111': -30, '020220222': 30, 
                      '001100': 12, '002200': -12, 
                      '001010': 12,'002020': -12, 
                      '000100': 2, '000200': -2,
                      #'10001': 8, '20002': -8, # 可解释为上述型
                      #'11001': 30, '22002': -30, #
                      #'10101': 30, '20202': -30, #
                      '211100': 30, '122200': -30, # 活三被拦阻后
                      '211010': 30, '122020': -30, # 
                      '210110': 30, '120220': -30, #
                      #'201110': 30, '102220': -30 #若2011100 按活三计
        }
        result = 0
        r_line=line[-1::-1]
        l = len(line)
        for span in (5,6):
            for k in range(l - span + 1):
                if line[k:k + span] in lines_dict:
                    result += lines_dict[line[k:k + span]]
                if r_line[k:k + span] in lines_dict:
                    result += lines_dict[r_line[k:k + span]]
        return result/2


    def evaluation(self, state: np.ndarray, cell_num=15) -> int: 
        """heuristic state evaluation"""
        result = 0
        # 行和列评分
        row_scores = np.array([self.score(''.join(map(str, row))) for row in state])
        col_scores = np.array([self.score(''.join(map(str, state[:, j]))) for j in range(cell_num)])
        result += row_scores.sum() + col_scores.sum()
        # 对角线（主对角线和副对角线）评分
        for offset in range(-cell_num + 1, cell_num):
            # 主对角线
            main_diag = ''.join(map(str, state.diagonal(offset)))
            result += self.score(main_diag)
            # 副对角线
            flipped_state = np.fliplr(state)  # 左右翻转
            anti_diag = ''.join(map(str, flipped_state.diagonal(offset)))
            result += self.score(anti_diag)
        return result
    
    def partial_score(self,r,c,state,player,path):
        """score of a position in one direction"""
        result = 0
        constant = 1.2  
        #assert constant>=1
        if state[r, c] == 0:
            dr, dc = path
            state[r, c] = 1  # 黑棋
            l_b = ''.join([str(state[r + l * dr, c + l * dc]) for l in range(-4, 5) if
                            Game.in_board(r + l * dr, c + l * dc, self.cell_num)])
            state[r, c] = 2  # 白棋
            l_w = ''.join([str(state[r + l * dr, c + l * dc]) for l in range(-4, 5) if
                            Game.in_board(r + l * dr, c + l * dc, self.cell_num)])
            state[r, c] = 0
            if player == 1:
                result += max(abs(self.score(l_b) * constant), abs(self.score(l_w)))  # policy1
                #result += self.score(l_b) * constant - self.score(l_w)               # policy2
            else:
                result += max(abs(self.score(l_b)), abs(self.score(l_w) * constant))  
                #result += self.score(l_b) - self.score(l_w) * constant 
        return result * self.base[r, c]

    def update_weight(self, r: int, c: int, state: np.ndarray, weight: np.ndarray):
        """update weight matrix"""
        weight[0, r, c] = 0
        weight[1, r, c] = 0
        temp = state[r,c]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for l in range(-4, 5):
                nr, nc = r + l * dr, c + l * dc
                if Game.in_board(nr, nc, self.cell_num) and state[nr, nc] == 0:
                    weight[0, nr, nc] += self.partial_score(nr, nc, state, 1, (dr,dc))
                    weight[1, nr, nc] += self.partial_score(nr, nc, state, 2, (dr,dc))
                    state[r,c]=0
                    weight[0, nr, nc] -= self.partial_score(nr, nc, state, 1, (dr,dc))
                    weight[1, nr, nc] -= self.partial_score(nr, nc, state, 2, (dr,dc))
                    state[r,c]=temp


    def greedy(self, weight: np.ndarray) -> tuple:
        """greedy policy"""
        goal = np.unravel_index(weight.argmax(), weight.shape)
        return goal
    
    
    def extend_frontier(self, weight:List[np.ndarray], player: int, num=0):
        """return the prior [num] coordinates"""
        len_available = np.count_nonzero(weight[player - 1])
        if num != 0:
            num = min(len_available, num)
        else:
            num = len_available
        flattened_matrix = weight[player - 1].flatten()
        top_k_indices = np.argsort(flattened_matrix)[-num:][::-1]
        rows, cols = np.unravel_index(top_k_indices, weight[player - 1].shape)
        top_k_coordinates = list(zip(rows, cols))
        return top_k_coordinates


    def place_piece(self, player: int, state: np.ndarray):
        """return a move"""
        target_x, target_y = self.cell_num // 2, self.cell_num // 2
        target_x, target_y = self.greedy(self.weight[player-1])
        if Game.is_valid(target_x, target_y, state, self.cell_num):
            return target_x, target_y, -1
        else:
            raise Exception(f'no valid move:({target_x},{target_y})')



class alphabetaplayer(Master):
    def __init__(self, cell_num: int, search_depth: int=2, search_num: int=2):
        super().__init__(cell_num)
        self.search_depth = search_depth
        self.option = search_num
    
    def switch_mode(self,depth,num):
        """set depth and branch size"""
        self.search_depth=depth
        self.option=num

    def place_piece(self, player: int, state: np.ndarray):
        """return a move"""
        target_x, target_y = self.cell_num // 2, self.cell_num // 2
        target_x, target_y = self.alpha_beta_search(player,state,self.weight)
        if Game.is_valid(target_x, target_y, state, self.cell_num):
            return target_x, target_y, -1
        else:
            raise Exception(f'no valid move:({target_x},{target_y})')  
    

    def alpha_beta_search(self, player: int, state: List[list], weight: List[list]):
        """alpha-beta prunning algorithm"""
        target_x, target_y = 0, 0
        alpha, beta = float('-inf'), float('inf')
        if player == 1:
            path = self.max_value(state, weight, alpha, beta, 1)[1]
        else:
            path = self.min_value(state, weight, alpha, beta, 1)[1]
        if len(path) > 0:
            target_x, target_y = path[0]
        if Game.is_valid(target_x, target_y, state, self.cell_num):
            return target_x, target_y
        else:
            raise Exception("Invalid position")


    def min_value(self, state: List[list], weight: np.ndarray, alpha: float, beta: float, depth: int):  # 白棋
        """min value node"""
        if depth > self.search_depth:
            temp=self.evaluation(state,self.cell_num)
            if abs(temp) > 200: #参数待定
                return self.cut_off_simulation(1,deepcopy(state),deepcopy(weight)), []  # eval
            else:
                return temp,[]
        else:
            max_num = max(self.option//(depth + 1), 4)
            frontier = self.extend_frontier(weight, 2, max_num)
            v, move, path = float('inf'), (-1, -1), []
            for x in frontier:
                e_x, e_y = x
                state[e_x, e_y] = 2
                if Game.success_evaluation(e_x, e_y, 2, state, self.cell_num):
                    utility = self.evaluation(state,self.cell_num)
                    state[e_x, e_y] = 0
                    return utility, [x]  # utility
                newweight = deepcopy(weight)
                self.update_weight(e_x, e_y, state, newweight)
                v2, temppath = self.max_value(state, newweight, alpha, beta, depth + 1)
                if v2 < v:
                    v = v2
                    move = x
                    path = [move] + temppath
                beta = min(beta, v)
                state[e_x, e_y] = 0
                if v <= alpha:
                    return v, path
            return v, path


    def max_value(self, state: np.ndarray, weight: np.ndarray, alpha: float, beta: float, depth: int):  # 黑棋
        """max value node"""
        if depth > self.search_depth:
            temp=self.evaluation(state,self.cell_num)
            if abs(temp) > 200: 
                return self.cut_off_simulation(1,deepcopy(state),deepcopy(weight)), [] 
            else:
                return temp,[]
        else:
            max_num = max(self.option// (depth + 1), 4)
            frontier = self.extend_frontier(weight, 1, max_num)
            v, move, path = float('-inf'), (-1, -1), []
            for x in frontier:
                e_x, e_y = x
                state[e_x,e_y] = 1
                if Game.success_evaluation(e_x, e_y, 1, state, self.cell_num):
                    utility = self.evaluation(state,self.cell_num)
                    state[e_x, e_y] = 0
                    return utility, [x]  # utility
                newweight = deepcopy(weight)
                self.update_weight(e_x, e_y, state, newweight)
                v2, temppath = self.min_value(state, newweight, alpha, beta, depth + 1)
                if v2 > v:
                    v = v2
                    move = x
                    path = [move] + temppath
                alpha = max(alpha, v)
                state[e_x, e_y] = 0
                if v >= beta:
                    return v, path
            return v, path
    

    def cut_off_simulation(self, player, state, weight, depth=0):
        """heuristic evaluation"""
        if depth > 4:
            return self.evaluation(state,self.cell_num)
        e_x,e_y=self.greedy(weight[player-1])
        state[e_x,e_y] = player
        if Game.success_evaluation(e_x, e_y, player, state, self.cell_num):
           return self.evaluation(state,self.cell_num)
        self.update_weight(e_x, e_y, state, weight)
        return self.cut_off_simulation(3 - player, state, weight, depth + 1)
    

    

class MCTSplayer(Master):

    def __init__(self, cell_num, n_rollout = 100):
        super().__init__(cell_num)
        self.n_rollout=n_rollout
        self.search_depth = 12
        self.option = 16
        self.root=MCT_node(np.zeros((self.cell_num, self.cell_num)), player=1, weight=self.weight) #当前根节点，用于追踪和重用树
    
    def switch_mode(self,depth,num,n_rollout):
        """set depth,branch size and rollout num"""
        self.search_depth=depth
        self.option=num
        self.n_rollout = n_rollout


    def place_piece(self, player: int, state: np.ndarray):
        """return a move and corresponding winrate"""
        target_x, target_y = self.cell_num // 2, self.cell_num // 2
        target_x, target_y, winrate = self.MCTS_pure(player,state,self.weight)
        if Game.is_valid(target_x, target_y, state, self.cell_num):
            return target_x, target_y, winrate
        else:
            raise Exception(f'no valid move:({target_x},{target_y})') 

        
    def MCTS_pure(self, player: int, state: np.ndarray, weight: np.ndarray):
        """pure MCTS algorithm"""
        root=MCT_node(state, player, weight)
        time_limit = 30
        start_time = time.time()
        while root.visits < self.n_rollout: # 约12s
            self.execute(root) 
            if time.time() - start_time > time_limit:
                break
        print(root.visits)
        # print([(x.get_move(), 0.5-0.5*x.Q, x.visits) for x in root.child], root.visits)
        choice:MCT_node= max(root.child, key=lambda x: x.visits)
        target_x,target_y=choice.get_move()
        winrate=0.5 + 0.5 * choice.Q
        # print('AI 胜率：{:.1f}%'.format(100 * winrate))
        return target_x, target_y, winrate
    
    def execute(self, node: MCT_node):
        """do one MC simulation"""
        while node.visits and not node.is_terminal:  # 逐层选择，直到未完全拓展的节点，返回它的一个新叶子
            node = self.select(node)
        if not node.is_terminal:
            result = self.rollout(node)
            node.backpropagate(result)
        else:
            node.backpropagate(1)
        
    def expand(self,node:MCT_node): #异步拓展， 每经过一次增加一个子节点，直到完全拓展
        """add a single child node"""
        if not node.childlist: # 第一次探索，获取所有可能的孩子节点
            node.childlist = self.extend_frontier(node.weight, node.player, self.option)
        action = node.childlist[len(node.child)] 
        e_x,e_y = action
        node.state[e_x,e_y] = node.player
        newweight = node.get_weight()
        self.update_weight(e_x, e_y, node.state, newweight)
        node.add_child(MCT_node(node.state, 3 - node.player, newweight , action , node, node.depth+1))
        if Game.success_evaluation(e_x, e_y, node.player, node.state, self.cell_num): 
            # 终止游戏：如果存在终态，则必然达到终态，不必继续扩展和探索这个节点 
            node.child[-1].is_terminal=True
            node.child[-1].backpropagate(1)
        node.state[e_x,e_y] = 0

    
    def rollout(self, node:MCT_node):
        """let's finish the game"""
        depth = node.depth
        player = node.player
        state  = node.get_state()
        weight = node.get_weight()
        while True:
            e_x,e_y = self.greedy(weight[player-1])
            state[e_x,e_y] = player
            if Game.success_evaluation(e_x, e_y, player, state, self.cell_num):
                break
            player = 3 - player
            self.update_weight(e_x, e_y, state, weight)
            depth += 1
            if depth > self.search_depth or self.cell_num**2 - np.count_nonzero(state) < 4:
                return 0        
        if player == node.player: 
            return -1
        else:
            return 1
        # 本节点的player表示即将落子的玩家，value评估是在上一层对手玩家视角进行的，如果本节点模拟结果player获胜，那么对上层节点就是一个坏结果，应该返回 -1 ????


    def select(self,node:MCT_node) -> MCT_node: # 由根至叶，选择没有经过rollout的节点
        """choose the child node to be explored"""
        if node.is_terminal:
            return node
        elif not node.is_fully_expanded(): # 没有完全拓展，增加子节点
            self.expand(node) # 增加一个子节点
            return node.child[-1] # 返回新增添的子节点
        else: #已经探索了全部子节点，按照PUCB方法选择
            return max(node.child, key=lambda x: x.get_value())
