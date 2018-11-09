#include <iostream>
#include <math.h>
#include <queue>
#include<map>

using namespace std;

#define len_row 4
#define len_col 4

int MDT[len_row][len_col];

//计算某个位置到最终结果的曼哈顿距离
int get_distance(int to_cal, int x, int y)
{
    return abs(ceil(float(to_cal)/len_col) - 1 - x) + abs((to_cal-1)%len_col - y); 
}

struct Puzzle{
    int graph[len_row][len_col];
    int zero_x, zero_y;
    int last_choice_x, last_choice_y;
    int total_distance;
    //计算整个图与目标图的距离差
    void get_total_distance(){
        total_distance = 0;
        for(int i = 0; i < len_row; i ++)
            for(int j = 0; j < len_col; j ++)
                if(graph[i][j] != 0) total_distance += get_distance(graph[i][j], i, j);
    }
    bool operator < (const Puzzle &a) const{
        return a.total_distance < total_distance;
    }
};

struct matrix{
    int graph[len_row][len_col];
};

void get_start(Puzzle& my_puzzle){
    for(int i = 0; i < len_row; i ++){
        for(int j = 0; j < len_col; j ++)
            if(my_puzzle.graph[i][j] == 0)
            {
                my_puzzle.zero_x = i;
                my_puzzle.zero_y = j;
            }
    }
}


bool illegal(int x, int y){ return(x < 0 || x >= len_row || y < 0 || y >= len_col); }
bool next_step(int, int, int, int, int, int, Puzzle);

void A_star_nextstep(Puzzle& tmp_state, priority_queue <Puzzle>& to_do, int x_choice, int y_choice)
{
    //走出边界
    if(illegal(tmp_state.zero_x + x_choice, tmp_state.zero_y + y_choice))
        return;
    //不走回头路
    if(x_choice + tmp_state.last_choice_x == 0 && y_choice + tmp_state.last_choice_y == 0)
        return;

    Puzzle next_to_do = tmp_state;
    next_to_do.zero_x = tmp_state.zero_x + x_choice;
    next_to_do.zero_y = tmp_state.zero_y + y_choice;
    swap(next_to_do.graph[tmp_state.zero_x][tmp_state.zero_y], next_to_do.graph[next_to_do.zero_x][next_to_do.zero_y]);
    next_to_do.last_choice_x = x_choice;
    next_to_do.last_choice_y = y_choice;
    //计算距离变化
    int dist1 = get_distance(next_to_do.graph[next_to_do.zero_x][next_to_do.zero_y], tmp_state.zero_x, tmp_state.zero_y);
    int dist2 = get_distance(next_to_do.graph[next_to_do.zero_x][next_to_do.zero_y], next_to_do.zero_x, next_to_do.zero_y);
    next_to_do.total_distance +=  dist1 - dist2;
    //if (next_to_do.total_distance < tmp_state.total_distance)
    to_do.push(next_to_do);
    return;
}

bool A_star(Puzzle my_puzzle)
{
    map<Puzzle, bool> state_map;

    priority_queue <Puzzle> to_do;
    my_puzzle.last_choice_x = 0, my_puzzle.last_choice_y = 0;
    state_map[my_puzzle] = true;
    cout << state_map[my_puzzle] << endl;
    to_do.push(my_puzzle);
    /*
    Puzzle tmp_state;
    //14 10 6 0 4 9 1 8 2 3 5 11 12 13 7 15
    int step = 0;
    while(!to_do.empty()){
        tmp_state = to_do.top();
        cout << "distance: " << tmp_state.total_distance << endl;
        cout << "size: " << to_do.size() << endl;
        for(int i = 0; i < len_row; i ++){
                for(int j = 0; j < len_col; j ++)
                    cout << tmp_state.graph[i][j] << ' ';
                cout << endl;
            }
        cout << endl;
        to_do.pop();
        if(tmp_state.total_distance == 0){
            for(int i = 0; i < len_row; i ++){
                for(int j = 0; j < len_col; j ++)
                    cout << tmp_state.graph[i][j] << ' ';
                cout << endl;
            }
            return true;
        }
        A_star_nextstep(tmp_state, to_do, 1, 0);
        A_star_nextstep(tmp_state, to_do, -1, 0);
        A_star_nextstep(tmp_state, to_do, 0, 1);
        A_star_nextstep(tmp_state, to_do, 0, -1);
    }
    */
    return false;
}

bool DFS(int depth, int x, int y, int last, Puzzle my_puzzle)
{
    //cout << my_puzzle.total_distance << endl;
    if(my_puzzle.total_distance == 0) 
    {
        //达到目标
        cout << "wtf?" << endl;
        for(int i = 0; i < len_row; i ++){
            for(int j = 0; j < len_col; j ++)
                cout << my_puzzle.graph[i][j] << ' ';
            cout << endl;
        }
        return true;
    }
    //IDA
    if(depth - my_puzzle.total_distance < 0){
        return false;
    }
    else{
        //不回头走
        if(last != 2)
            if( next_step(depth - 1, x, y, x + 1, y, 1, my_puzzle) )
                return true;
        if(last != 1)
            if( next_step(depth - 1, x, y, x - 1, y, 2, my_puzzle) )
                return true;
        if(last != 4)
            if( next_step(depth - 1, x, y, x, y + 1, 3, my_puzzle) )
                return true;
        if(last != 3)
            if( next_step(depth - 1, x, y, x, y - 1, 4, my_puzzle) )
                return true;
        return false;
    }
}

bool next_step(int depth, int x, int y, int next_x, int next_y, int current, Puzzle my_puzzle){
    if(illegal(next_x, next_y))
        return false;
    Puzzle tmp = my_puzzle;
    //算总距离的变化
    my_puzzle.total_distance += get_distance(my_puzzle.graph[next_x][next_y], x, y) - get_distance(my_puzzle.graph[next_x][next_y], next_x, next_y);
    swap(my_puzzle.graph[x][y], my_puzzle.graph[next_x][next_y]);
    if(DFS(depth, next_x, next_y, current, my_puzzle))
        return true;
    my_puzzle = tmp;
    return false;
}


//终止条件为总的曼哈顿距离为0
//每次移动之后不会回到上一次的状态
//11 3 1 7 4 6 8 2 15 9 10 13 14 12 5 0
//14 10 6 0 4 9 1 8 2 3 5 11 12 13 7 15
//1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0
//0 5 15 14 7 9 6 13 1 2 12 10 8 11 4 3
//6 10 3 15 14 8 7 11 5 1 0 2 13 12 9 4
int main()
{
    clock_t start,end;
    start = clock();

    Puzzle my;
    for(int i = 0; i < len_row; i ++){
        for(int j = 0; j < len_col; j ++)
            cin >> my.graph[i][j];
    }
    get_start(my);
    my.get_total_distance();
    cout << my.total_distance << endl;
    /*
    // IDA

    for(int i = my.total_distance; i < 100; i ++){
        Puzzle tmp = my;
        if(DFS(i, my.zero_x, my.zero_y, -1, tmp)){
            cout << "solution found" << endl;
            break;
        }
    }
    */
    if(A_star(my))
        cout << "solution found" << endl;
    end = clock();
    double dur = (double)(end - start);
    cout << "Time used: " << (dur/CLOCKS_PER_SEC);
}