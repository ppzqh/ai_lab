'''
**TODO**
    定义get_next_pos作为搜索算法，调用形式如下：

        pos_i, pos_j = get_next_pos(current_node)

    输入为当前节点，输出为搜索到的坐标

'''

import numpy as np
from DQN import DeepQNetwork
from DQN import my_state
'''
定义常量
'''
N = 8
start_status = np.zeros((8,8), dtype=int)
start_status[3,3], start_status[4,4] = -1, -1
start_status[3,4], start_status[4,3] = 1, 1
BLACK_CHESS ='\u25CF' #●
WHITE_CHESS ='\u25CB' #○
CLICK_CHESS ='\u00D7'
VALID_CHESS ='\u2605'
chess_dict = {1:BLACK_CHESS, -1:WHITE_CHESS,
              0:CLICK_CHESS, 2:VALID_CHESS}

'''
定义节点，包括函数
    1.get_valid_pos -- 计算可走的位置
    2.move -- 在指定位置落子
    3.show -- 显示棋盘
    4.get_valid_pos_ij -- 函数1的辅助函数
'''
class State:
    def __init__(self, status=start_status, turn=1):
        self.status = status.copy()
        np.place(self.status, self.status==2, 0)
        self.turn = turn #1表示轮到黑方
        self.valid_pos = []
        self.get_valid_pos()
    
    def get_valid_pos(self):
        [self.get_valid_pos_ij(i,j) for i in range(N) for j in range(N) if self.status[i][j] == self.turn]
        
    def get_valid_pos_ij(self, i, j):
        #水平方向
        if (j < N - 2 and self.status[i][j+1] == -self.turn):
            k = j + 2
            while k < N and self.status[i][k] == -self.turn: k += 1
            if k < N and self.status[i][k] == 0:
                self.status[i][k] = 2
                self.valid_pos.append([i,k])
        if (j > 1 and self.status[i][j-1] == -self.turn):
            k = j - 2
            while k >= 0 and self.status[i][k] == -self.turn: k -= 1
            if k >= 0 and self.status[i][k] == 0:
                self.status[i][k] = 2
                self.valid_pos.append([i,k])
        #竖直方向
        if (i < N - 2 and self.status[i+1][j] == -self.turn):
            k = i + 2
            while k < N and self.status[k][j] == -self.turn: k += 1
            if k < N and self.status[k][j] == 0:
                self.status[k][j] = 2
                self.valid_pos.append([k,j])
        if (i > 1 and self.status[i-1][j] == -self.turn):
            k = i - 2
            while k >= 0 and self.status[k][j] == -self.turn: k -= 1
            if k >= 0 and self.status[k][j] == 0:
                self.status[k][j] = 2
                self.valid_pos.append([k,j])
        #右下对角线方向
        if (j < N - 2 and i < N - 2 and self.status[i+1][j+1] == -self.turn):
            k = 2
            while j + k < N and i + k < N and self.status[i+k][j+k] == -self.turn: k += 1
            if j + k < N and i + k < N and self.status[i+k][j+k] == 0:
                self.status[i+k][j+k] = 2
                self.valid_pos.append([i+k,j+k])
        if (j > 1 and i > 1 and self.status[i-1][j-1] == -self.turn):
            k = 2
            while j - k >= 0 and i - k >= 0 and self.status[i-k][j-k] == -self.turn: k += 1
            if j - k >= 0 and i - k >= 0 and self.status[i-k][j-k] == 0:
                self.status[i-k][j-k] = 2
                self.valid_pos.append([i-k,j-k])
        #左下对角线方向
        if (i < N - 2 and j > 1 and self.status[i+1][j-1] == -self.turn):
            k = 2
            while i + k < N and j - k >= 0 and self.status[i+k][j-k] == -self.turn: k += 1
            if i + k < N and j - k >= 0 and self.status[i+k][j-k] == 0:
                self.status[i+k][j-k] = 2
                self.valid_pos.append([i+k,j-k])
        if (j < N - 2 and i > 1 and self.status[i-1][j+1] == -self.turn):
            k = 2
            while j + k < N and i - k >= 0 and self.status[i-k][j+k] == -self.turn: k += 1
            if j + k < N and i - k >= 0 and self.status[i-k][j+k] == 0:
                self.status[i-k][j+k] = 2
                self.valid_pos.append([i-k,j+k])
    
    def move(self, i, j):
        self.status[i][j] = self.turn
        #水平方向
        k = j + 1
        while k < N and self.status[i][k] == -self.turn: k += 1
        if k < N and self.status[i][k] == self.turn:
            for w in range(j + 1, k): self.status[i][w] = self.turn
        k = j - 1
        while k >= 0 and self.status[i][k] == -self.turn: k -= 1
        if k >= 0 and self.status[i][k] == self.turn:
            for w in range(k + 1, j): self.status[i][w] = self.turn
        #竖直方向
        k = i + 1
        while k < N and self.status[k][j] == -self.turn: k += 1
        if k < N and self.status[k][j] == self.turn:
            for w in range(j + 1, k): self.status[w][j] = self.turn
        k = i - 1
        while k >= 0 and self.status[k][j] == -self.turn: k -= 1
        if k >= 0 and self.status[k][j] == self.turn:
            for w in range(k + 1, i): self.status[w][j] = self.turn
        #右下对角线方向
        k = 1
        while j + k < N and i + k < N and self.status[i+k][j+k] == -self.turn: k += 1
        if j + k < N and i + k < N and self.status[i+k][j+k] == self.turn:
            for w in range(1,k): self.status[i+w][j+w] = self.turn
        k = 1
        while j - k >= 0 and i - k >= 0 and self.status[i-k][j-k] == -self.turn: k += 1
        if j - k >= 0 and i - k >= 0 and self.status[i-k][j-k] == self.turn:
            for w in range(1, k): self.status[i-w][j-w] = self.turn
        #左下对角线方向
        k = 1
        while j + k < N and i - k >= 0 and self.status[i-k][j+k] == -self.turn: k += 1
        if j + k < N and i - k >= 0 and self.status[i-k][j+k] == self.turn:
            for w in range(1,k): self.status[i-w][j+w] = self.turn
        k = 1
        while j - k >= 0 and i + k < N and self.status[i+k][j-k] == -self.turn: k += 1
        if j - k >= 0 and i + k < N and self.status[i+k][j-k] == self.turn:
            for w in range(1, k): self.status[i+w][j-w] = self.turn
    
    def show(self):
        print('  ', end='')
        for i in range(N): print(i, end='  ')
        print()
        for i in range(N):
            print(i, end=' ')
            for j in range(N):
                print(chess_dict[self.status[i][j]], end=' ')
            print('')

    def isTerminal(self):
        end = False
        if len(self.valid_pos) == 0:
            np.place(self.status, self.status==2, 0)
            self.turn = -self.turn #换个持方
            self.get_valid_pos()
            if len(self.valid_pos) == 0:
                end = True
            else:
                np.place(self.status, self.status==2, 0)
            self.turn = -self.turn
        return end
            
'''
获取玩家输入落子坐标
'''
def get_input(valid_pos):
    while 1:
        try:
            a, b = input('请输入坐标，如 >>> {} {} '.format(valid_pos[0][0], valid_pos[0][1])).split()
            if a == 'q': 
                print('不玩了')
                break
            a, b = int(a), int(b)
            if [a,b] in valid_pos:
                return a, b
            else:
                print('这个走不了哇')
        except Exception as e:
            print(str(e))
            pass


'''
游戏开始函数
'''
def Start_game(AI):
    print('============================')
    print('    黑白棋游戏(黑方先手)')
    print('============================')
    player = input('请选择是否先手？（Y/N）')
    player = 1 if player == 'y' or player == 'Y' else -1
    cur_state = State(status=start_status, turn=1)
    no_way = 0 #记录无子可走的次数
    while 1:
        black, white = (cur_state.status == 1).sum().sum(), (cur_state.status == -1).sum().sum()
        print('============================')
        print('    当前状态--轮到{0}方'.format('黑' if cur_state.turn==1 else '白'))
        print('    黑子：{0}，白子：{1}'.format(black,white))
        print('============================')
        cur_state.show()
        if black + white == N * N: break
        if len(cur_state.valid_pos) == 0: 
            print('当前方没有可走的棋，跳过当前轮')
            if no_way == 1: break
            no_way = 1
        else:
            no_way = 0
            print('可走的棋有：', cur_state.valid_pos)
            if cur_state.turn == player: #玩家顺序
                pos_i, pos_j = get_input(cur_state.valid_pos)
            else:
                print('电脑计算中...')
                pos_i, pos_j = get_next_pos(cur_state, AI)
                print('电脑计算结果：{} {}'.format(pos_i, pos_j))
            #移动
            cur_state.move(pos_i, pos_j)
        cur_state = State(cur_state.status, -cur_state.turn)
    black, white = (cur_state.status == 1).sum().sum(), (cur_state.status == -1).sum().sum()
    print('============================')
    print('          结束状态')
    print('    黑子：{0}，白子：{1}'.format(black,white))
    print('============================')
    cur_state.show()

'''
搜索算法，返回搜索结果坐标
'''

def start_DQN_game(AI1, AI2):
    player = 1
    cur_state = State(status=start_status, turn=1)
    no_way = 0 #记录无子可走的次数

    #全局变量记录最后一个状态
    tmp_state = None
    last_pos_i, last_pos_j = None, None
    while 1:
        black, white = (cur_state.status == 1).sum().sum(), (cur_state.status == -1).sum().sum()
        if black + white == N * N: 
            #black, white = (cur_state.status == 1).sum().sum(), (cur_state.status == -1).sum().sum()
            result = black - white
            AI1.store_transition(tmp_state, cur_state.status, 1 if result > 0 else 0, last_pos_i*N+last_pos_j)
            AI2.store_transition(tmp_state, cur_state.status, 0 if result > 0 else 1, last_pos_i*N+last_pos_j)
            break
        if len(cur_state.valid_pos) == 0: 
            #print('当前方没有可走的棋，跳过当前轮')
            if no_way == 1:
                #black, white = (cur_state.status == 1).sum().sum(), (cur_state.status == -1).sum().sum()
                result = black - white
                AI1.store_transition(tmp_state, cur_state.status, 1 if result > 0 else 0, last_pos_i*N+last_pos_j)
                AI2.store_transition(tmp_state, cur_state.status, 0 if result > 0 else 1, last_pos_i*N+last_pos_j)
                break
            no_way = 1
        else:
            no_way = 0
            # 训练时不显示
            # print('可走的棋有：', cur_state.valid_pos)
            if cur_state.turn == player: #玩家顺序
                #pos_i, pos_j = get_input(cur_state.valid_pos)
                #print('电脑1计算中...')
                pos_i, pos_j = get_next_pos(cur_state, AI1)
                #print('电脑1计算结果：{} {}'.format(pos_i, pos_j))
            else:
                #print('电脑2计算中...')
                pos_i, pos_j = get_next_pos(cur_state, AI2)
                #print('电脑2计算结果：{} {}'.format(pos_i, pos_j))
            #移动
            tmp_state = cur_state.status.copy()
            last_pos_i, last_pos_j = pos_i, pos_j
            cur_state.move(pos_i, pos_j)

        cur_state = State(cur_state.status, -cur_state.turn)

def get_next_pos(cur_state, dqn:DeepQNetwork):
    action_values = dqn.choose_action(cur_state.status)
    if (len(action_values) == 0):
        return cur_state.valid_pos[np.random.randint(0, len(cur_state.valid_pos))]
    x,y = [None,None]

    for i in cur_state.valid_pos:
        tmp_reward = action_values[0][i[0]*N + i[1]]
        if x == None or tmp_reward > action_values[0][x*N + y]:
            x,y = i[0],i[1]
    return [x,y]

def trainDQN():
    AI1 = DeepQNetwork()
    AI2 = DeepQNetwork()
    for episode in range(500000):
        start_DQN_game(AI1, AI2)
        if episode % 5 == 0:
            AI1.learn()
            AI2.learn()
        print("episode:", episode)
    AI1.save_model()
    AI2.save_model()
trainDQN()
'''
AI1 = DeepQNetwork()
AI1.load_model()
Start_game(AI1)
'''