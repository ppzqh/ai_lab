#include <iostream>
using namespace std;
#include <vector>
#include <set>
#include <map>

#define len_row 6
#define len_col 6

bool illegal(int x, int y){
    return x < 0 || x >= len_row || y < 0 || y >= len_col;
}
bool gameOver(char graph[len_row][len_col]){
    for(int i = 0; i < len_row; i ++)
        for(int j = 0; j < len_col; j ++){
            if(graph[i][j] == '*')
                return false;
        }
    return true;
}
char initGraph[len_row][len_col] = 
{
    {'*', '*', '*', '*', '*', '*'},
    {'*', '*', '*', '*', '*', '*'},
    {'*', '*', 'X', 'O', '*', '*'},
    {'*', '*', 'O', 'X', '*', '*'},
    {'*', '*', '*', '*', '*', '*'},
    {'*', '*', '*', '*', '*', '*'},
};

char initGraph2[len_row][len_col] = 
{
    {'*', '*', '*', '*', '*', '*'},
    {'*', '*', '*', '*', '*', '*'},
    {'*', '*', 'O', 'X', '*', '*'},
    {'*', '*', 'X', 'O', '*', '*'},
    {'*', '*', '*', '*', '*', '*'},
    {'*', '*', '*', '*', '*', '*'},
};

struct reverseStruct{
    int startX, startY;
    int endX, endY;
    reverseStruct(int sx, int sy, int ex, int ey):startX(sx), startY(sy), endX(ex), endY(ey){}
    bool operator <(const reverseStruct& a) const{
        return startX < a.startX || startY < a.startY || endX < a.endX || endY < a.endY;
    }
    bool operator ==(const reverseStruct& a) const{
        return endX == a.endX && endY == a.endY;
    }
};

struct Matrix{
    char graph[len_row][len_col];
    int white, black;
    Matrix(){}
    Matrix(char initGraph[len_row][len_col]){
        white = 0, black = 0;
        for(int i = 0; i < len_row; i ++)
            for(int j = 0; j < len_col; j ++){
                graph[i][j] = initGraph[i][j];
                white += initGraph[i][j] == 'O';
                black += initGraph[i][j] == 'X';
            }
    }

    void printGraph(){
        for(int i = 0; i < len_row; i ++){
            for(int j = 0; j < len_col; j ++)
                cout << graph[i][j] << ' ';
            cout << endl;
        }
        cout << endl;
    }
   //action表示是reverse还是search, reverse为1，search为0
    void search(int x, int y, set<reverseStruct>& to_do, char type){
        for(int i = -1; i <= 1; i ++){
            for(int j = -1; j <= 1; j ++){
                if(i != 0 || j != 0)
                    get(x, y, i, j, to_do, type);
            }
        }
    }
    void get(int x, int y, int _x, int _y, set<reverseStruct>& to_do, char type){
        char tmpType = type == 0 ? 'O':'X';
        char opponent = tmpType == 'O' ? 'X' : 'O';
        int tmpX = x, tmpY = y;
        do{
            tmpX += _x;
            tmpY += _y;
        }while(!illegal(tmpX, tmpY) && graph[tmpX][tmpY] == opponent);
        //将可以选择的位置放入to_do中
        if(!illegal(tmpX, tmpY) && graph[tmpX][tmpY] != tmpType && (abs(tmpX - x) >= 2 || abs(tmpY - y) >= 2) ){
            to_do.insert(reverseStruct(x, y, tmpX, tmpY));
        }
    }

    void nextStep(bool type){
        char charType = type == 0 ? 'O':'X';
        set<reverseStruct> to_do;
        for(int i = 0; i < len_row; i ++){
            for(int j = 0; j < len_col; j ++){
                if(graph[i][j] == charType)
                    search(i, j, to_do, type);
            }
        }
        /*输出可以走的位置
        for(set<reverseStruct>::iterator iter = to_do.begin(); iter != to_do.end(); iter ++){
            cout << iter->endX << ' ' << iter->endY << endl;
        }
        */
        if(to_do.size()){
            if(type) black += 1;
            else white += 1;
            reverseAll(to_do);
        }
    }

    void reverse(int startX, int startY, int endX, int endY){
        //翻转函数，从Start到End中间的全部翻转
        int _x = endX - startX > 0 ? 1 : endX - startX == 0 ? 0 : -1;
        int _y = endY - startY > 0 ? 1 : endY - startY == 0 ? 0 : -1;

        graph[endX][endY] = graph[startX][startY];
        //更新黑白子的个数
        bool tmpType = graph[startX][startY] == 'X' ? 1 : 0;
        int toAdd = abs(startX - endX) == 0 ? abs(startY - endY) : abs(startX - endX);
        if(tmpType){
            white -= (toAdd - 1);
            black += (toAdd - 1);
        }
        else{
            white += (toAdd - 1);
            black -= (toAdd - 1);
        }
        while(startX + _x != endX || startY + _y != endY){
            graph[startX + _x][startY + _y] = graph[endX][endY];
            startX += _x;
            startY += _y;
        }
    }
    void reverseAll(set<reverseStruct>& to_do){
        //记录走到每一个位置所得到的效益（这里是由吃子个数决定的）
        map<pair<int,int>, int> score;
        for(set<reverseStruct>::iterator iter = to_do.begin(); iter != to_do.end(); iter++){
            int tmpDistance = abs(iter->endX - iter->startX) + abs(iter->endY - iter->startY);
            score[pair<int,int>(iter->endX,iter->endY)] += (tmpDistance - 1);
        }
        //取得最大的
        map<pair<int,int>, int>::const_iterator it = score.end();
        it --;
        //cout << it->first.first << endl;
        for(set<reverseStruct>::iterator iter = to_do.begin(); iter != to_do.end(); iter++){
            if(iter->endX == it->first.first && iter->endY == it->first.second)
                reverse(iter->startX, iter->startY, iter->endX, iter->endY);
        }
        printGraph();
    }
    //去重
    bool equal(const Matrix& a){
        bool e1 = 1, e2 = 1, e3 = 1;
        for(int i = 0; i < len_row; i ++){
            for(int j = 0; j < len_col; j ++){
                //逆时针旋转90度: X2 + y1 = len - 1 & x1 = y2 中心对称: x1 + x2 = len -1
                if(graph[i][j] != a.graph[len_row - 1 - j][i]) e1 = false; 
                if(graph[i][j] != a.graph[j][len_row - 1 - i]) e2 = false;
                if(graph[i][j] != a.graph[len_row - 1 - i][len_row - 1 - j]) e3 = false;
                if(!e1 && !e2 && !e3) return false;
            }
        }
        return true;
    }
};

struct Node{
    Matrix m;
    //0代表'O'子， 1代表‘X’子
    bool type;
    //0代表极小节点，1代表极大节点
    int score;
    vector<Node*> children;
    Node(Matrix m1, bool type):m(m1), type(type){}
};

struct Tree{
    Node* root;
    int depth;
    Tree(Node* root, int depth):root(root), depth(depth){}
    void TreeInsert(Node* tmpRoot){
        if(tmpRoot->children.size()){
            for(int index = 0; index < tmpRoot->children.size(); index ++)
                TreeInsert(tmpRoot->children[index]);
        }
        else{
            bool nextType = !tmpRoot->type;
            Node* toInsert = new Node(tmpRoot->m, nextType);
            toInsert->m.nextStep(nextType);
            //去重
            bool isEqual = true;
            for(int i = 0; i < tmpRoot->children.size(); i ++){
                if(toInsert->m.equal(tmpRoot->children[i]->m))
                    isEqual = false;
            }
            if(isEqual) tmpRoot->children.push_back(toInsert);
        }
    }
    /*
    void insertLayer(){

    }
    */
    void printTree(Node* tmpRoot){
        cout << tmpRoot->type << endl;
        if(tmpRoot->children.size()){
            for(int index = 0; index < tmpRoot->children.size(); index ++){
                //cout << tmpRoot->children[index]->score;
                printTree(tmpRoot->children[index]);
            }
        }
    }
};

int main(){
    Matrix m = Matrix(initGraph);
    int step = 0;
    while(1){
        cout << "O turn" << endl;
        m.nextStep(0);
        cout << "black: " << m.black << " white: " << m.white << endl;
        if(gameOver(m.graph))
            break;

        cout << "X turn" << endl;
        m.nextStep(1);
        cout << "black: " << m.black << " white: " << m.white << endl;
        if(gameOver(m.graph)) break;
        step ++;
    }
    cout << "step:" << step << endl;
    /*
    Tree puzzle(new Node(Matrix(initGraph), 0), 2);
    int step = 40;
    while(step --){
        puzzle.TreeInsert(puzzle.root);
    }
    */
    //puzzle.printTree(puzzle.root);
}