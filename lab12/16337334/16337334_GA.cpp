#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <limits.h>
#include <fstream>

using namespace std;

#define N 8

int Start_Map[N][N] = {
	{0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0},
	{0,0,0,-1,1,0,0,0},
	{0,0,0,1,-1,0,0,0},
	{0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0}
};

vector<int> G_weight(16,1);
/*
int Start_Map[N][N] = {
	{0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0},
	{0,0,1,1,1,0,0,0},
	{0,0,0,1,1,1,0,0},
	{0,-1,-1,-1,1,1,1,0},
	{0,0,0,-1,0,1,0,0},
	{0,0,1,-1,-1,-1,-1,-1},
	{0,0,0,0,0,0,0,0}
};
*/

string tochar(int a){
	if (a == 1) return "●";
	if (a == -1) return "○";
	if (a == 0) return "×";
	return "  ";
}

struct State
{
	int MAP[N][N];
	int turn; //turn为1表示轮到黑方
	vector<int> next_i;
	vector<int> next_j;
	/*实现alpha-beta所需属性*/
	vector<State*> child;
	int alpha, beta;
	int H;

	State(int M[N][N]=Start_Map, int t=1,int a=INT_MIN, int b=INT_MAX, 
		int pos_i=-1, int pos_j=-1,bool show_pos=true):turn(t),alpha(a),beta(b){
		set_MAP(M);
		//若pos_j不为0，表示上一个State落子在(pos_i, pos_j)位置
		if (pos_j != -1) {turn = -t; move(pos_i, pos_j); turn = t;}
		get_next_pos(show_pos);
		child = vector<State*>(next_i.size(), NULL);
	}

	//按照给定权重矩阵计算分数
	int calc_H(vector<int>& weight){
		int white_score = 0, black_score = 0;
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++){
					int a = (i < N/2? i: N - 1 - i);
					int b = (j < N/2? j: N - 1 - j);
				if (MAP[i][j] == -1) 
					white_score += weight[a * N/2 + b];
				else if (MAP[i][j] == 1) 
					black_score += weight[a * N/2 + b];
			}
		return black_score - white_score;
	}

	void set_MAP(int M[N][N]){
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				this->MAP[i][j] = M[i][j] == 2? 0: M[i][j];
	}

	void get_next_pos(bool show_pos=true){
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				if (this->MAP[i][j] == this->turn) get_next_pos_with_ij(i, j, show_pos);
	}

	void get_next_pos_with_ij(int i, int j, bool show_pos){
		//水平方向
		if (j < N - 2 && MAP[i][j+1] == -turn){
			int k = j + 2;
			while (k < N && MAP[i][k] == -turn) k++;
			if (k < N && MAP[i][k] == 0){
				if (show_pos == true) MAP[i][k] = 2; 
				next_i.push_back(i); next_j.push_back(k);}
		}
		if (j > 1 && MAP[i][j-1] == -turn){
			int k = j - 2;
			while (k >= 0 && MAP[i][k] == -turn) k--;
			if (k >= 0 && MAP[i][k] == 0){
				if (show_pos == true) MAP[i][k] = 2;
				next_i.push_back(i); next_j.push_back(k);}
		}
		//竖直方向
		if (i < N - 2 && MAP[i+1][j] == -turn){
			int k = i + 2;
			while (k < N && MAP[k][j] == -turn) k++;
			if (k < N && MAP[k][j] == 0) {
				if (show_pos == true) MAP[k][j] = 2; 
				next_i.push_back(k); next_j.push_back(j);}
		}
		if (i > 1 && MAP[i-1][j] == -turn){
			int k = i - 2;
			while (k >= 0 && MAP[k][j] == -turn) k--;
			if (k >= 0 && MAP[k][j] == 0) {
				if (show_pos == true) MAP[k][j] = 2; 
				next_i.push_back(k); next_j.push_back(j);}
		}
		//右下对角线
		if (j < N - 2 && i < N - 2 && MAP[i+1][j+1] == -turn){
			int k = 2;
			while (j + k < N && i + k < N && MAP[i+k][j+k] == -turn) k++;
			if (j + k < N && i + k < N && MAP[i+k][j+k] == 0){
				if (show_pos == true) MAP[i+k][j+k] = 2; 
				next_i.push_back(i+k); next_j.push_back(j+k);}
		}
		if (j > 1 && i > 1 && MAP[i-1][j-1] == -turn){
			int k = 2;
			while (j - k >= 0 && i - k >= 0 && MAP[i-k][j-k] == -turn) k++;
			if (j - k >= 0 && i - k >= 0 && MAP[i-k][j-k] == 0){
				if (show_pos == true) MAP[i-k][j-k] = 2; 
				next_i.push_back(i-k); next_j.push_back(j-k);}
		}
		//左下对角线
		if (i < N - 2 && j > 1 && MAP[i+1][j-1] == -turn){
			int k = 2;
			while(i+k < N && j - k >= 0 && MAP[i+k][j-k] == -turn) k++;
			if (i+k < N && j - k >= 0 && MAP[i+k][j-k] == 0){
				if (show_pos == true) MAP[i+k][j-k] = 2; 
				next_i.push_back(i+k); next_j.push_back(j-k);}
		}
		if (j < N - 2 && i > 1 && MAP[i-1][j+1] == -turn){
			int k = 2;
			while(j+k < N && i - k >= 0 && MAP[i-k][j+k] == -turn) k++;
			if (j+k < N && i - k >= 0 && MAP[i-k][j+k] == 0){
				if (show_pos == true) MAP[i-k][j+k] = 2; 
				next_i.push_back(i-k); next_j.push_back(j+k);}
		}
	}

	void move(int i, int j){
		MAP[i][j] = turn;
		int k;
		//水平方向
		k = j + 1;
		while(k < N && MAP[i][k] == -turn) k++;
		if (k < N && MAP[i][k] == turn) 
			for (int w = j + 1; w < k; w++) MAP[i][w] = turn;
		k = j - 1;
		while(k >= 0 && MAP[i][k] == -turn) k--;
		if (k >= 0 && MAP[i][k] == turn) 
			for (int w = j - 1; w > k; w--) MAP[i][w] = turn;
		//竖直方向
		k = i + 1;
		while (k < N && MAP[k][j] == -turn) k++;
		if (k < N && MAP[k][j] == turn)
			for (int w = i + 1; w < k; w++) MAP[w][j] = turn;
		k = i - 1;
		while (k >= 0 && MAP[k][j] == -turn) k--;
		if (k >= 0 && MAP[k][j] == turn)
			for (int w = i - 1; w > k; w--) MAP[w][j] = turn;
		//右下对角线
		k = 1;
		while (j + k < N && i + k < N && MAP[i+k][j+k] == -turn) k++;
		if (j + k < N && i + k < N && MAP[i+k][j+k] == turn)
			for (int w = 1; w < k; w++) MAP[i+w][j+w] = turn;
		k = 1;
		while (j - k >= 0 && i - k >= 0 && MAP[i-k][j-k] == -turn) k++;
		if (j - k >= 0 && i - k >= 0 && MAP[i-k][j-k] == turn)
			for (int w = 1; w < k; w++) MAP[i-w][j-w] = turn;
		//左下对角线
		k = 1;
		while(i+k < N && j - k >= 0 && MAP[i+k][j-k] == -turn) k++;
		if (i+k < N && j - k >= 0 && MAP[i+k][j-k] == turn)
			for (int w = 1; w < k; w++) MAP[i+w][j-w] = turn;
		k = 1;
		while(j+k < N && i - k >= 0 && MAP[i-k][j+k] == -turn) k++;
		if (j+k < N && i - k >= 0 && MAP[i-k][j+k] == turn)
			for (int w = 1; w < k; w++) MAP[i-w][j+w] = turn;
	}

	void show_map(){
		
		cout << "            ";
		for (int i = 0; i < N; i++) cout << i << "  ";
		for (int i = 0; i < N; i++){
			cout << "\n          " << i;
			for (int j = 0; j < N; j++){
				cout << ' ' << tochar(MAP[i][j]);
			}
		}
		cout << endl;
	}

	void get_chess_num(int & black, int & white){
		black = white = 0;
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				if (MAP[i][j] == 1) black++;
				else if (MAP[i][j] == -1) white++;
	}



	bool isTerminal(){
		bool end = false;
		if (next_i.size() == 0){
			set_MAP(MAP);
			turn = -turn;
			get_next_pos(false);
			if (next_i.size() == 0) end = true;
		}
		return end;
	}
};

/*alpha-beta剪枝*/
void make_tree_with_depth(State* start, int& next_move, vector<int>&weight, int length=0, int depth=2){
	//turn为1表示下一步走黑棋
	if (length >= depth || start->next_i.size() == 0 || start->beta <= start->alpha){
		start->H = start->calc_H(weight);
		start->alpha = max(start->alpha, start->H);
		start->beta = min(start->beta, start->H);
		return;
	}
	int next_move_tmp = -3;
	for (unsigned int i = 0; i < start->next_i.size(); i++){
		int next_x = start->next_i[i];
		int next_y = start->next_j[i];
		State * next = new State(start->MAP, -start->turn, start->alpha, 
								 start->beta, next_x, next_y);

		start->child[i] = next;
		make_tree_with_depth(next, next_move_tmp, weight, length+1, depth);
		if (start->turn == 1 && next->beta > start->alpha)
			{start->alpha = next->beta; next_move = i;}
		else if (start->turn == -1 && next->alpha < start->beta)
			{start->beta = next->alpha; next_move = i;}
		delete next;
	}

}

//显示DNA
void show_gene(vector<vector<int> >& gene_list){
	for (int i = 0; i < (int)gene_list.size(); i++){
		cout << "GENE " << i << " >>>" << endl;
		for (int j = 0; j < (N * N / 4); j++){
			cout << gene_list[i][j] << ' ';
			if (j % 4 == 3) cout << endl;
		}
		cout << endl;
	}
}

//随机初始化种群
void init_gene(vector<vector<int> >& gene_list){
	for (int i = 0; i < (int)gene_list.size(); i++)
		for (int j = 0; j < (N * N / 4); j++)
			gene_list[i][j] = 1 + rand() % 50;
}

//在竞争时双方根据DNA选择移动位置，规则自定
void get_move(State * A, vector<int>& gene_list, int & pos_i, int & pos_j){
	//将当前基因表作为alpha-beta的权重表，搜索进行选择
	int next_move = -1;
	make_tree_with_depth(A, next_move, gene_list, 0, 4);
	pos_i = A->next_i[next_move];
	pos_j = A->next_j[next_move];
}

/*遗传算法从这里开始*/
//竞争，终局评价函数自定
int competition(vector<int>& player1, vector<int>& player2){
	State *A = new State(), *temp = NULL;
	int black, white;
	int player = 1; //是否为黑方
	int No_way = 0;
	int pos_i = -1, pos_j = -1;
	while(1){
		A->get_chess_num(black, white);
		if (black + white == 64) break;
		if (A->next_i.size() == 0){
			//当前方没有可走的棋，跳过当前轮
			if (++No_way == 2)//双方均没有可走的棋，结束
				break;
		}
		else{
			No_way = 0;
			//player1顺序
			if (A->turn == player) get_move(A, player1, pos_i, pos_j);
			//player2顺序
			else get_move(A, player2, pos_i, pos_j);
			A->move(pos_i, pos_j);
		}
		temp = A;
		A = new State(A->MAP, -A->turn);
		delete temp;
		temp = NULL;
	}
	A->get_chess_num(black, white);
	delete A;
	return (black >= white? 1: -1); //返回player1是否胜利
}

//在区间内随机选择一个基因
int random_gene(int start, int end){
    return rand() % (end - start) + start;
}

//选择出适应度高的个体
void select(vector<vector<int> >& gene_list){
	int index1, index2;
    //记录对弈得分
    vector<int> tmp_score_list(gene_list.size(), 0);
    //竞争20（个体总数）次
	for (int i = 0; i < gene_list.size(); i++){
        //每次随机选择2个个体，先后手各对弈一局
        index1 = random_gene(0, gene_list.size());
        do{
            index2 = random_gene(0, gene_list.size());
        }while(index1 == index2);

		int tmp_score = 0;
		//alpha-beta，记录先后手对局结果
		tmp_score += competition(gene_list[index1], gene_list[index2]);
		tmp_score -= competition(gene_list[index2], gene_list[index1]);

        //记录双方得分
        tmp_score_list[index1] += tmp_score;
        tmp_score_list[index2] -= tmp_score;
	}
    //根据得分将基因表排序
    for (int i = 0; i < gene_list.size(); i ++){
        for (int j = 0; j < gene_list.size() - i - 1; j ++){
            if(tmp_score_list[j] > tmp_score_list[j + 1]){
                swap(tmp_score_list[j], tmp_score_list[j+1]);
                swap(gene_list[j], gene_list[j+1]);
            }
        }
    }
    return;
}

//交叉
void crossover(vector<vector<int> >& gene_list){
	int index1, index2;
	for (int i = gene_list.size() / 2; i < gene_list.size(); i++){
		/*从得分排名一半之前选择两个基因进行交叉*/
		index1 = random_gene(gene_list.size() / 2, gene_list.size());
        do{
            index2 = random_gene(gene_list.size() / 2, gene_list.size());
        }while(index1 == index2);
        //对于选择的两个基因，比较两者的得分，利用得分调整交叉替换的概率。
		for (int j = 0; j < N * N / 4; j++){
			int tmp = rand() % gene_list.size();
            int half = gene_list.size() / 2;
            //比较得分来确定交叉策略
            //index是排好序的，越大的证明得分越高，排序越高被选中的概率越高
            if(index1 < index2) 
                gene_list[index1][j] = (tmp >  half + index2 - index1 ? gene_list[index1][j]: gene_list[index2][j]);
            else 
                gene_list[index2][j] = (tmp >  half + index1 - index2 ? gene_list[index1][j]: gene_list[index2][j]);
		}
	}
    return;
}

//变异
void mutation(vector<vector<int> >& gene_list){
    int size = gene_list.size();
    int half = gene_list.size() / 2;
	for (int i = 0; i < half; i++){
		/*将排序前一半的基因进行变异*/
		int index = rand() % half + half;
		for (int j = 0; j < N * N / 4; j++){
			int tmp = rand() % size;
			//替换的概率为 10 + i2 - i1 / 20
			gene_list[i][j] = gene_list[index][j];
			if (tmp <= half + index - i) 
                gene_list[i][j] += (rand() % 20 - 10); 
            //判断是否越界
			gene_list[i][j] = gene_list[i][j] <= 0 ? 1 : gene_list[i][j];
            gene_list[i][j] = gene_list[i][j] >= 50 ? 49 : gene_list[i][j];
		}
	}
}

//遗传算法
void GA(vector<vector<int> > &gene_list, int epoch){
	init_gene(gene_list);
	int i = 0;
	while (i ++ <= epoch){
		select(gene_list);
		crossover(gene_list);
		mutation(gene_list);
        //输出epoch
        cout << "Epoch " << i << '/' << epoch << ' ' << endl;
        //输出基因集合 show_DNAs(gene_list);
	}
}

//游戏进行函数
void Game_start(){
	State *A = new State();
	State *temp = NULL;
	int pos_i = -1, pos_j = -1;
	int black, white;
	int player = 1; //表示玩家为黑方
	int No_way = 0;
	char choose;
	cout << "              ==================" << endl;
	cout << "             黑白棋游戏(黑方先手)" << endl;
	cout << "              ==================" << endl;
	cout << "             请选择是否先手(Y/N):";
	cin >> choose;
	cin.clear();
	cin.sync();
	if (choose != 'Y' && choose != 'y') player = 0; 

	while(1){
		A->get_chess_num(black, white);
		cout << "              ==================" << endl;
		cout << "              当前状态"<< "(轮到" << (A->turn? "黑":"白") << "方)" << endl;
		cout << "               黑子：" << black << " 白子：" << white << endl;
		cout << "              ==================" << endl;
		A->show_map();
		if (black + white == 64) break;
		if (A->next_i.size() == 0){
			cout << " 当前方没有可走的棋，跳过当前轮" << endl;
			if (++No_way == 2){
				cout << "              ==================" << endl;
				cout << "              双方均没有可走的棋，结束" << endl;
				cout << "              ==================" << endl;
				break;
			}
		}
		else{
			No_way = 0;
			cout << " 可走的棋为：";
			for (unsigned int i = 0; i < A->next_i.size(); i++)
				cout << '(' << A->next_i[i] << ',' << A->next_j[i] << ") ";
			//玩家顺序
			if (A->turn == player){
				cout << " 请输入落子的坐标(例: "<< A->next_i[0] << ' ' << A->next_j[0] << ")：";
				cin >> pos_i >> pos_j;
				while(pos_i >= N || pos_i < 0 || pos_j >= N || pos_j < 0 || A->MAP[pos_i][pos_j] != 2){
					cin.clear();
					cin.sync();
					cout << "\n 不是合法的坐标，请重新输入(例: "<< A->next_i[0] << ' ' << A->next_j[0] << ")：";
					cin >> pos_i >> pos_j;
				}
				cin.clear();
				cin.sync();
			}
			//电脑顺序
			else{
				int next_move = -1;
				cout << "\n 电脑计算中..." << endl;
				make_tree_with_depth(A, next_move, G_weight, 0, 5);
				pos_i = A->next_i[next_move];
				pos_j = A->next_j[next_move];
				cout << " 电脑计算结果：选择走(" << pos_i << ',' << pos_j << ")\n";
				cout << "           H,alpha,beta=" << A->H << ',' << A->alpha << ',' << A->beta << endl;
			}
			A->move(pos_i, pos_j);
		}
		temp = A;
		A = new State(A->MAP, -A->turn, INT_MIN, INT_MAX);
		delete temp;
		temp = NULL;
	}
	A->get_chess_num(black, white);
	cout << "              ==================" << endl;
	cout << "                   结束状态" << endl;
	cout << "               黑子：" << black << " 白子：" << white << endl;
	cout << "              ==================" << endl;
	A->show_map();
}


int main(int argc, char const *argv[]){
    //遗传算法训练权重矩阵
	srand(time(0));
	vector<vector<int> > gene_list(20, vector<int>(16,0));
	GA(gene_list, 1000);
    ofstream outfile;
    outfile.open("data.txt");
    for(int i = 0; i < 20; i ++){
        outfile << i << endl;
        for(int j = 0; j < 16; j ++)
            outfile << gene_list[i][j] << ' ';   
        outfile << endl; 
    }
    outfile.close();
}