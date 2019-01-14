class VariableElimination:
    @staticmethod
    def inference(factor_list, query_variables,
                  ordered_list_of_hidden_variables, evidence_list):
        for ev in evidence_list:
            # Your code here
            tmp_list = factor_list.copy()
            for i in range(len(factor_list)):
                if ev in factor_list[i].var_list:
                    tmp_list.append(factor_list[i].restrict(ev, evidence_list[ev]))
                    tmp_list.remove(factor_list[i])
            factor_list = tmp_list
                
        for var in ordered_list_of_hidden_variables:
            # Your code here
            first = False
            tmp_list = factor_list.copy()
            for i in range(len(factor_list)):
                if var in factor_list[i].var_list:
                    tmp_list.remove(factor_list[i])
                    if not first:
                        new_factor = factor_list[i] 
                        first = True
                    else: new_factor = new_factor.multiply(factor_list[i])
            tmp_list.append(new_factor.sum_out(var))
            factor_list = tmp_list

        print("RESULT: ")
        res = factor_list[0]
        for factor in factor_list[1:]:
            res = res.multiply(factor)
        total = sum(res.cpt.values())
        res.cpt = {k: v / total for k, v in res.cpt.items()}
        res.print_inf()

    @staticmethod
    def print_factors(factor_list):
        for factor in factor_list:
            factor.print_inf()


class Util:
    @staticmethod
    def to_binary(num, len):
        return format(num, '0' + str(len) + 'b')


class Node:
    def __init__(self, name, var_list):
        self.name = name
        self.var_list = var_list
        self.cpt = {}

    def set_cpt(self, cpt):
        self.cpt = cpt

    def print_inf(self):
        print("Name = " + self.name)
        print(" vars " + str(self.var_list))
        for key in self.cpt:
            print("   key: " + key + " val : " + str(self.cpt[key]))
        print()

    def same(self, f1, f2, list1, list2):
        for i, j in zip(list1, list2):
            if f1[i] != f2[j]:
                return False
        return True

    def multiply(self, factor):
        '''function that multiplies with another factor'''
        # Your code here
        common = [i for i in self.var_list if i in factor.var_list]
        if not len(common): return self

        index1, index2 = [self.var_list.index(i) for i in common], [factor.var_list.index(i) for i in common]
        new_list = self.var_list + [i for i in factor.var_list if i not in self.var_list]
        new_cpt = {}
        for i in self.cpt:
            for j in factor.cpt:
                if self.same(i, j, index1, index2):
                    jStr = ''
                    for s in range(len(j)):
                        if s not in index2: jStr += j[s]
                    new_cpt[i + jStr] = self.cpt[i] * factor.cpt[j]
        new_node = Node('f' + str(new_list), new_list)
        new_node.set_cpt(new_cpt)
        return new_node

    def sum_out(self, variable):
        '''function that sums out a variable given a factor'''
        # Your code here
        new_var_list = [j for j in self.var_list if j != variable]
        var_index = self.var_list.index(variable)
        new_cpt = {}
        for i in self.cpt:
            new_cpt[i[:var_index] + i[var_index+1:]] = new_cpt.get(i[:var_index] + i[var_index+1:],0) + self.cpt[i]
        new_node = Node('f' + str(new_var_list), new_var_list)
        new_node.set_cpt(new_cpt)
        return new_node

    def restrict(self, variable, value):
        '''function that restricts a variable to some value
        in a given factor'''
        # Your code here
        new_var_list = [j for j in self.var_list if j != variable]
        var_index = self.var_list.index(variable)
        new_cpt = {}
        for i in self.cpt:
            if i[var_index] == chr(value + ord('0')):
                new_cpt[i[:var_index] + i[var_index+1:]] = self.cpt[i]
        new_node = Node('f' + str(new_var_list), new_var_list)
        new_node.set_cpt(new_cpt.copy())
        return new_node


# Create nodes for Bayes Net
B = Node('B', ['B'])
E = Node('E', ['E'])
A = Node('A', ['A', 'B', 'E'])
J = Node('J', ['J', 'A'])
M = Node('M', ['M', 'A'])

# Generate cpt for each node
B.set_cpt({'0': 0.999, '1': 0.001})
E.set_cpt({'0': 0.998, '1': 0.002})
A.set_cpt({'111': 0.95, '011': 0.05, '110': 0.94, '010': 0.06,
           '101':0.29, '001': 0.71, '100': 0.001, '000': 0.999})
J.set_cpt({'11': 0.9, '01': 0.1, '10': 0.05, '00': 0.95})
M.set_cpt({'11': 0.7, '01': 0.3, '10': 0.01, '00': 0.99})

print("P(A) **********************")
VariableElimination.inference([B, E, A, J, M], ['A'], ['B', 'E', 'J', 'M'], {})

print("P(B | J, ~M) **********************")
VariableElimination.inference([B, E, A, J, M], ['B'], ['E', 'A'], {'J':1, 'M':0})
