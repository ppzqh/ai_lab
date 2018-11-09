from queue import Queue
import numpy as np 
import time

read_file = open('MazeData.txt').readlines()
maze = list()

def illegal(x, y, maze, visit):
    '''
    x,y表示当前节点的坐标
    '''
    #判断节点是否在迷宫范围内，且可以访问（即迷宫中为0且未访问过）
    if x < 0 or x >= len(maze) or y < 0 or y >= len(maze[0]):
        return False
    elif maze[x][y] != 1 and visit[x][y] == 0:
        return True
    return False

def BFS_print_path(parent_dict, path, start, end):
    tmp_node = end
    while(tmp_node != start):
        path[tmp_node[0]][tmp_node[1]] = 1
        tmp_node = parent_dict[tmp_node]
    print_path(path)

def BFS(start, end, maze, visit):
    '''
    start:起点
    end:终点
    maze:地图
    visit:标记访问过的节点的矩阵
    '''
    q = Queue()
    q.put(start)

    parent_dict = {}
    path = np.zeros( (len(maze), len(maze[0])) )
    while not q.empty():
        head = q.get()
        #找到终点
        if head == end:
            print("exit found")
            BFS_print_path(parent_dict, path, start, end)
            return
        #向四个邻居扩展，若邻居可扩展，即加入队列
        if illegal(head[0] - 1, head[1], maze, visit):
            q.put( (head[0] - 1, head[1]) )
            parent_dict[(head[0] - 1, head[1])] = head
            visit[head[0] - 1][head[1]] = 1

        if illegal(head[0] + 1, head[1], maze, visit):
            q.put( (head[0] + 1, head[1]) )
            parent_dict[(head[0] + 1, head[1])] = head
            visit[head[0] + 1][head[1]] = 1

        if illegal(head[0], head[1] - 1, maze, visit):
            q.put( (head[0], head[1] - 1) )
            parent_dict[(head[0], head[1] - 1)] = head
            visit[head[0]][head[1] - 1] = 1

        if illegal(head[0], head[1] + 1, maze, visit):
            q.put( (head[0], head[1] + 1) )
            parent_dict[(head[0], head[1] + 1)] = head
            visit[head[0]][head[1] + 1] = 1

def print_path(visit):
    for i in visit:
        tmp_str = ''
        for j in i:
            tmp_str += str(int(j))
        print(tmp_str)


def iterative_deepening_search(x, y, end, depth, maze, visit):
#指定depth，限制深度优先搜索的深度。
    if (x, y) == end:
        print('exit found')
        print_path(visit)
        return True
    else:
        if depth < 0:
            return False
        #向四个邻居扩展
        if illegal(x + 1, y, maze, visit):
            #标记扩展的邻居
            visit[x+1][y] = 1
            if iterative_deepening_search(x+1, y, end, depth-1, maze, visit):
                return True
            #回溯
            visit[x+1][y] = 0
        if illegal(x - 1, y, maze, visit):
            visit[x-1][y] = 1
            if iterative_deepening_search(x-1, y, end, depth-1, maze, visit):
                return True
            visit[x-1][y] = 0
        if illegal(x, y + 1, maze, visit):
            visit[x][y + 1] = 1
            if iterative_deepening_search(x, y + 1, end, depth-1, maze, visit):
                return True
            visit[x][y + 1] = 0
        if illegal(x, y - 1, maze, visit):
            visit[x][y - 1] = 1
            if iterative_deepening_search(x, y - 1, end, depth-1, maze, visit):
                return True
            visit[x][y - 1] = 0
        return False

'''main'''
#get maze
for row_index, row in enumerate(read_file):
    tmp_row = list()
    for column_index, i in enumerate(row.rstrip('\n')):
        if i != 'S' and i != 'E':
            tmp_row.append(int(i))
        else:
            if i == 'S': 
                start = (row_index, column_index)
            else:
                end = (row_index, column_index)
            tmp_row.append(-1) 
    maze.append(tmp_row)

visit = np.zeros( (len(maze), len(maze[0])) )
visit[start[0]][start[1]] = 1     

'''
#iterative deepening search
start_time = time.clock() 
elapsed = (time.clock() - start_time)
depth = 1
while not iterative_deepening_search(start[0], start[1], end, depth, maze, visit):
    depth += 1
print(depth)
start_time = time.clock()  
iterative_deepening_search(start[0], start[1], end, depth, maze, visit)
print("iterative deepening search time used:",elapsed)
'''
#BFS
start_time = time.clock()
BFS(start, end, maze, visit)
elapsed = (time.clock() - start_time)
print("BFS time used:",elapsed)
