#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
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
void show_DNAs(vector<vector<int> >& DNAs){
	for (int i = 0; i < (int)DNAs.size(); i++){
		cout << "DNA " << i << " >>>" << endl;
		for (int j = 0; j < (N * N / 4); j++){
			cout << DNAs[i][j] << ' ';
			if (j % 4 == 3) cout << endl;
		}
		cout << endl;
	}
}

//随机初始化种群
void initialize(vector<vector<int> >& DNAs){
	for (int i = 0; i < (int)DNAs.size(); i++)
		for (int j = 0; j < (N * N / 4); j++)
			DNAs[i][j] = 1 + rand() % 50;
}

//在竞争时双方根据DNA选择移动位置，规则自定
void get_move(State * A, vector<int>& DNA, int & pos_i, int & pos_j){
	/*直接选择最大的
	int max_w = 0, max_i = 0;
	for (unsigned int i = 0; i < A->next_i.size(); i++){
		int a = (A->next_i[i] < N / 2? A->next_i[i]: N - 1 - A->next_i[i]);
		int b = (A->next_j[i] < N / 2? A->next_j[i]: N - 1 - A->next_j[i]);
		if (DNA[a * N / 2 + b] > max_w){
			max_w = DNA[a * N / 2 + b];
			max_i = i;
		}
	pos_i = A->next_i[max_i];
	pos_j = A->next_j[max_i];
	}
	*/

	/*按照权重随机选择，权重大的选中可能性越大
	int temp[A->next_i.size() + 1] = {0};
	for (unsigned int i = 0; i < A->next_i.size(); i++){
		int a = (A->next_i[i] < N / 2? A->next_i[i]: N - 1 - A->next_i[i]);
		int b = (A->next_j[i] < N / 2? A->next_j[i]: N - 1 - A->next_j[i]);
		temp[i + 1] = temp[i] + DNA[a * N / 2 + b];
	}
	int choice = rand() % temp[A->next_i.size()];
	unsigned int i = 0;
	for (; i < A->next_i.size(); i++)
		if (choice <= temp[i + 1]) break;
	pos_i = A->next_i[i];
	pos_j = A->next_j[i];
	*/
	//alpha-beta剪枝根据权重进行选择
	int next_move = -1;
	make_tree_with_depth(A, next_move, DNA, 0, 4);
	pos_i = A->next_i[next_move];
	pos_j = A->next_j[next_move];
}

//竞争，终局评价函数自定
int fight(vector<int>& player1, vector<int>& player2){
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

//在[start, end)随机选择两个下标进行竞争
void random_choose_DNA(int& index1, int& index2, int start, int end){
	index1 = rand() % (end - start) + start;
	do {
		index2 = rand() % (end - start) + start;
	}while(index1 == index2);
}

//选择出适应度高的个体
int selection(vector<vector<int> >& DNAs){
	/*要改成两两对局积分，还没改
	vector<int> fitness(DNAs.size(),0);
	for (unsigned int i = 0; i < DNAs.size(); i++){
		for(unsigned int j = 0; j < DNAs.size(); j++)
			if (i != j) fitness[i] += fight(DNAs[i], DNAs[j]);
	}
	*/
	
	//每次随机选择2个个体，各先后手对局5次
	int index1, index2;
	int score = 0;
	for (int i = 0; i < (int)DNAs.size() / 2; i++){
		random_choose_DNA(index1, index2, i, DNAs.size());
		//简单选择策略，随机选两个进行竞争，赢的留下
		int player1_win = 0;
		//使用alpha-beta时只需要打一局，结果是固定的
		player1_win += fight(DNAs[index1], DNAs[index2]);
		player1_win -= fight(DNAs[index2], DNAs[index1]);
		// for (int r = 0; r < 5; r++) player1_win += fight(DNAs[index1], DNAs[index2]);
		// for (int r = 0; r < 5; r++) player1_win -= fight(DNAs[index2], DNAs[index1]);
		// cout << "Competition between DNA " << index1 << " and " << index2 << ">>>";
		// cout << "DNA " << index1 << " gets score: " << player1_win << endl;
		score += abs(player1_win);
		if (player1_win)swap(DNAs[i], DNAs[index1]);
		else swap(DNAs[i], DNAs[index2]);
	}
	return score;
}

//交叉，自定
void corssover(vector<vector<int> >& DNAs){
	int index1, index2;
	for (unsigned int i = DNAs.size() / 2; i < DNAs.size() - 5; i++){
		/*随机选择两个selection中选择的DNA做均匀交叉*/
		random_choose_DNA(index1, index2, 0, (int)DNAs.size() / 2);
		for (int j = 0; j < N * N / 4; j++){
			int tmp = rand() % 2;
			DNAs[i][j] = (tmp? DNAs[index1][j]: DNAs[index2][j]);
		}
	}
}

//变异，自定
void mutation(vector<vector<int> >& DNAs){
	for (int i = 0; i < 5; i++){
		/*随机选择前面selection和crossover的DNA做均匀变异*/
		int index = rand() % (DNAs.size() - 5);
		for (int j = 0; j < N * N / 4; j++){
			int r = rand() % 10;
			//替换的概率为6/10
			DNAs[DNAs.size() - i - 1][j] = DNAs[index][j];
			if (r <= 6) DNAs[DNAs.size() - i - 1][j] += (rand() % 20 - 10); 
			if (DNAs[DNAs.size() - i - 1][j] <= 0)DNAs[DNAs.size() - i - 1][j] = 1; 
			if (DNAs[DNAs.size() - i - 1][j] > 49)DNAs[DNAs.size() - i - 1][j] = 49; 
		}
	}
}

//遗传算法
void genetic_algorithm(vector<vector<int> > &DNAs, int epoch = 100){
	initialize(DNAs);
	int i = 0;
	while (i < epoch){
		int score_sum = 0;
		score_sum += selection(DNAs);
		corssover(DNAs);
		mutation(DNAs);
		
		if (i % 2 == 0){
			//这个输出没用嘤嘤嘤
			printf("Epoch %d/%d >>> score_sum = %d/20\n", i, epoch, score_sum);
			show_DNAs(DNAs);
		}
		i++;
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
	//Game_start(); //游戏开始
    
   //遗传算法训练权重
	srand(time(0));
	vector<vector<int> > DNAs(20, vector<int>(16,0));
	genetic_algorithm(DNAs, 20000);
    
	// srand(time(0));
	// vector<vector<int> > DNAs(10, vector<int>(16,0));
	// initialize(DNAs);
	// show_DNAs(DNAs);
	// selection(DNAs);
	// show_DNAs(DNAs);
	// corssover(DNAs);
	// show_DNAs(DNAs);
	// mutation(DNAs);
	// show_DNAs(DNAs);
}