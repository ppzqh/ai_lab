#include <iostream>
#include <math.h>
using namespace std;
#include <vector>

#define len 13

int total = 0;
void printQueen(int puzzle[len]){
    for(int i = 0; i < len; i ++){
        for(int j = 0; j < len; j ++){
            if(j == puzzle[i]) cout << 'Q' << ' ';
            else cout << '*' << ' ';
        }
        cout << endl;
    }
}
/*
检测方法
判断当前状态是否合法，即是否符合N皇后问题的规则。
1.每一行有且仅有一个皇后（这里不可能出现）
2.每一列有且仅有一个皇后
3.每一条对角线有且仅有一个皇后
*/
bool check(int puzzle[len], int num){
   for(int i = 0; i < num - 1; i ++){
        if(puzzle[i] == puzzle[num - 1] || abs(puzzle[i] - puzzle[num - 1]) == abs(i - num + 1))
            return false;
   }
    return true;
}

/*
puzzle：一个一位数组，每一位对应一行，其中的值代表对应行的皇后的列序号
num：代表当前搜索的行序号
*/
void search(int puzzle[len], int num){
    //判断当前状态是否合法
    if(!check(puzzle, num)) return;
    //搜索到最底端，结束
    if(num == len){
        ++ total;
        // if(total == 1) printQueen(puzzle);
        return;
    }
    //对于当前行，搜索所有摆放位置
    for(int i = 0; i < len; i ++){
        //直接赋值，之后判断
        puzzle[num] = i;
        search(puzzle, num + 1);
    }
}

/*
domain:该行的能摆放的位置的值域，一开始全部置为-1.
num:该行能摆放位置的数量
*/
struct Queen{
    int domain[len];
    int num;
    Queen():num(len){
        for(int i = 0; i < len; i ++)
            domain[i] = -1;
    }

};

/*
row, col:当前皇后的行列位置
对row之后的所有行进行值域缩减。
*/
bool fcCheck(int row, int col, vector<Queen>& CurDom){
    //若该行已经被放置皇后，则不继续搜索
    if(CurDom[row].domain[col] != -1) return false;
    //对当前行之后的所有行的值域进行改变
    for(int i = 1; i < len - row; i ++){
        //与当前皇后同一列的位置都删除
        if(CurDom[row + i].domain[col] == -1){
            CurDom[row + i].domain[col] = row;
            -- CurDom[row + i].num;
        }
        //与当前皇后同一对角线的位置都删除
        if(col - i >= 0 && CurDom[row + i].domain[col - i] == -1){
            CurDom[row + i].domain[col - i] = row;
            -- CurDom[row + i].num;
        }
        if(col + i < len && CurDom[row + i].domain[col + i] == -1){
            CurDom[row + i].domain[col + i] = row;
            -- CurDom[row + i].num;
        }
        //如果这一行的值域为空，则说明该解不正确，停止搜索
        if(!CurDom[row + i].num)
            return false;
    }
    return true;
}

void FC(vector<Queen >& CurDom, int num){
    //到达底端，结束搜索
    if(num == len){
        ++ total;
        return;
    }
    //对当前行所有的位置进行搜索
    for(int i = 0; i < len; i ++){
        //fcCheck中对值域进行改变，如果check返回真则可以继续下一行的搜索
        if(fcCheck(num, i, CurDom)) FC(CurDom, num + 1);
        
        //回溯，对fcCheck中的改变进行恢复
        for(int j = 1; j < len - num; j ++){
            if(CurDom[num + j].domain[i] == num){
                CurDom[num + j].domain[i] = -1;
                ++ CurDom[num + j].num;
            }
            if(i - j >= 0 && CurDom[num + j].domain[i - j] == num){
                CurDom[num + j].domain[i - j] = -1;
                ++ CurDom[num + j].num;
            }
            if(i + j < len && CurDom[num + j].domain[i + j] == num){
                CurDom[num + j].domain[i + j] = -1;
                ++ CurDom[num + j].num;
            }
        }
    }
}

int main(){
    cout << "N: " << len << endl;
    clock_t start,end;
    //回溯
    start = clock();
    int puzzle[len] = {-1}; 
    search(puzzle, 0);
    cout << "num of solutions: " << total << endl;

    end = clock();
    cout << "回溯: " << (double)(end - start)/CLOCKS_PER_SEC << endl;
    
    //前向检测
    total = 0;
    start = clock();
    vector<Queen> CurDom(len);
    FC(CurDom, 0);
    cout << "num of solutions: " << total << endl;
    end = clock();
    cout << "前向检测: " << (double)(end - start)/CLOCKS_PER_SEC << endl;
}
