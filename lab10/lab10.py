#TASK1
import pomegranate as pg

guestDistribution = pg.DiscreteDistribution( {'A':1./3, 'B':1./3, 'C':1./3} )
prizeDistribution = pg.DiscreteDistribution( {'A':1./3, 'B':1./3, 'C':1./3} )

table = []
for prize in ['A', 'B', 'C']:
    for guest in ['A', 'B', 'C']:
        for monty in ['A', 'B', 'C']:
            tmpPro = 0.0 if(monty == guest or monty == prize) else 0.5 if(guest == prize) else 1.0 
            table.append([prize, guest, monty, tmpPro])

montyCondition = pg.ConditionalProbabilityTable(table, [guestDistribution, prizeDistribution])
s1 = pg.Node(guestDistribution, name='guest')
s2 = pg.Node(prizeDistribution, name='prize')
s3 = pg.Node(montyCondition, name='monty')

model = pg.BayesianNetwork('Monty Hall Problem')
model.add_nodes(s1, s2, s3)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.bake()
print(model.probability(['A','C','B']))
print(model.probability(['A','C','A']))

#TASK2
#TASK2
Burglary = pg.DiscreteDistribution({'B':0.001, '~B':0.999})
Earthquake = pg.DiscreteDistribution({'E':0.002, '~E':0.998})
Alarm = pg.ConditionalProbabilityTable([
    ['B', 'E', 'A', 0.95],
    ['B', 'E', '~A', 0.05],

    ['B', '~E', 'A', 0.94],
    ['B', '~E', '~A', 0.06],

    ['~B', 'E', 'A', 0.29],
    ['~B', 'E', '~A', 0.71],

    ['~B', '~E', 'A', 0.001],
    ['~B', '~E', '~A', 0.999]], [Burglary, Earthquake]
)

John = pg.ConditionalProbabilityTable([
    ['A', 'J', 0.90],
    ['A', '~J', 0.10],
    ['~A', 'J', 0.05],
    ['~A', '~J', 0.95]], [Alarm]
)

Mary = pg.ConditionalProbabilityTable([
    ['A', 'M', 0.70],
    ['A', '~M', 0.30],
    ['~A', 'M', 0.01],
    ['~A', '~M', 0.99]], [Alarm]
)

task2_model = pg.BayesianNetwork('task2')
s_b = pg.Node(Burglary, name='burglary')
s_e = pg.Node(Earthquake, name='earthquake')
s_a = pg.Node(Alarm, name='alarm')
s_j = pg.Node(John, name='john')
s_m = pg.Node(Mary, name='mary')

task2_model.add_nodes(s_b, s_e, s_a, s_j, s_m)
task2_model.add_edge(s_b, s_a)
task2_model.add_edge(s_e, s_a)
task2_model.add_edge(s_a, s_j)
task2_model.add_edge(s_a, s_m)
task2_model.bake()

def getProbability(model, eventList, index, toDo, domain_list):
    if not len(eventList): return 0
    if(index == len(eventList)):
        return model.probability(toDo)
    result = 0
    for i in domain_list[eventList[index]]:
        toDo[eventList[index]] = i
        result += getProbability(model, eventList, index+1, toDo, domain_list)
    return result

b_domain = ['B','~B']
e_domain = ['E','~E']
a_domain = ['A','~A']
j_domain = ['J','~J']
m_domain = ['M','~M']
domain_list = [b_domain, e_domain, a_domain, j_domain, m_domain]
def getEventList(toDo):
    eventList = []
    for i in range(len(toDo)):
        if toDo[i] == '':
            eventList.append(i)
    return eventList

toDo = ['', '', '', 'J', 'M']
eventList = getEventList(toDo)
result1 = getProbability(task2_model, eventList, 0, toDo, domain_list)
result2 = task2_model.probability(['B', 'E', 'A', 'J', 'M'])
print(result1)
print(result2)


toDo = ['', '', 'A', 'J', 'M']
eventList = getEventList(toDo)
result3 = getProbability(task2_model, eventList3, 0, toDo, domain_list) / result1
print(result3)

eventList4 = [1, 2]
toDo4 = ['~B', '', '', 'J', '~M']
result4 = getProbability(task2_model, eventList4, 0, toDo4, domain_list) / 0.999

eventList = [1,2,3,4]
toDo = ['~B','','','','']
result1 = getProbability(task2_model, eventList, 0, toDo, domain_list)  

eventList = [1, 2]
toDo = ['~B','','','J','~M']
result = getProbability(task2_model, eventList, 0, toDo, domain_list) / result1
print(result)

#TASK3
patientAge = pg.DiscreteDistribution({'0-30':0.10, '31-65':0.30, '65+':0.60})
CTScanResult = pg.DiscreteDistribution({'IS':0.7, 'HS':0.3})
MRIScanResult = pg.DiscreteDistribution({'IS':0.7, 'HS':0.3})
Anticoagulants = pg.DiscreteDistribution({'U':0.5, 'N':0.5})

StrokeType = pg.ConditionalProbabilityTable([
    ['IS','IS','IS', 0.8],
    ['IS','HS','IS', 0.5],
    ['HS','IS','IS', 0.5],
    ['HS','HS','IS', 0.0],

    ['IS','IS','HS', 0.0],
    ['IS','HS','HS', 0.4],
    ['HS','IS','HS', 0.4],
    ['HS','HS','HS', 0.9],

    ['IS','IS','SM', 0.2],
    ['IS','HS','SM', 0.1],
    ['HS','IS','SM', 0.1],
    ['HS','HS','SM', 0.1]], [CTScanResult, MRIScanResult]
)

#Mortality
Mortality = pg.ConditionalProbabilityTable([
    ['IS', 'U', 'F', 0.28],
    ['HS', 'U', 'F', 0.99],
    ['SM', 'U', 'F', 0.1],

    ['IS', 'N', 'F', 0.56],
    ['HS', 'N', 'F', 0.58],
    ['SM', 'N', 'F', 0.05],

    ['IS', 'U', 'T', 0.72],
    ['HS', 'U', 'T', 0.01],
    ['SM', 'U', 'T', 0.9],

    ['IS', 'N', 'T', 0.44],
    ['HS', 'N', 'T', 0.42],
    ['SM', 'N', 'T', 0.95]], [StrokeType, Anticoagulants]
)

#disability
Disability = pg.ConditionalProbabilityTable([
    ['IS', '0-30', 'Neg', 0.80],
    ['HS', '0-30', 'Neg', 0.70],
    ['SM', '0-30', 'Neg', 0.9],
    ['IS', '31-65', 'Neg', 0.60],
    ['HS', '31-65', 'Neg', 0.50],
    ['SM', '31-65', 'Neg', 0.4],
    ['IS', '65+', 'Neg', 0.30],
    ['HS', '65+', 'Neg', 0.20],
    ['SM', '65+', 'Neg', 0.1],

    ['IS', '0-30', 'Mod', 0.10],
    ['HS', '0-30', 'Mod', 0.20],
    ['SM', '0-30', 'Mod', 0.05],
    ['IS', '31-65', 'Mod', 0.3],
    ['HS', '31-65', 'Mod', 0.40],
    ['SM', '31-65', 'Mod', 0.3],
    ['IS', '65+', 'Mod', 0.40],
    ['HS', '65+', 'Mod', 0.20],
    ['SM', '65+', 'Mod', 0.1],

    ['IS', '0-30', 'Sev', 0.10],
    ['HS', '0-30', 'Sev', 0.10],
    ['SM', '0-30', 'Sev', 0.05],
    ['IS', '31-65', 'Sev', 0.10],
    ['HS', '31-65', 'Sev', 0.10],
    ['SM', '31-65', 'Sev', 0.3],
    ['IS', '65+', 'Sev', 0.30],
    ['HS', '65+', 'Sev', 0.60],
    ['SM', '65+', 'Sev', 0.8]], [StrokeType, patientAge]
)

n1 = pg.Node(patientAge)
n2 = pg.Node(CTScanResult)
n3 = pg.Node(MRIScanResult)
n4 = pg.Node(StrokeType)
n5 = pg.Node(Anticoagulants)
n6 = pg.Node(Mortality)
n7 = pg.Node(Disability)

task3_model = pg.BayesianNetwork()
task3_model.add_nodes(n1,n2,n3,n4,n5,n6,n7)

task3_model.add_edge(n2, n4)
task3_model.add_edge(n3, n4)

task3_model.add_edge(n1, n6)
task3_model.add_edge(n4, n6)

task3_model.add_edge(n4, n7)
task3_model.add_edge(n5, n7)

task3_model.bake()
p_domain = ['0-30', '31-65', '65+']
c_domain = ['IS', 'HS']
m_domain = ['IS', 'HS']
s_domain = ['IS', 'HS', 'SM']
a_domain = ['U', 'N']
mo_domain = ['T', 'F']
d_domain = ['Neg', 'Mod', 'Sev']

domain_list = [p_domain, c_domain, m_domain, s_domain, a_domain, mo_domain, d_domain]
#p1 = P(Mortality='True' | PatientAge='0-30' , CTScanResult='Ischemic Stroke')
toDo = ['0-30', 'IS', '', '', '', 'T', '']
eventList = getEventList(toDo)
result1 = getProbability(task3_model, eventList, 0, toDo, domain_list)
toDo = ['0-30', 'IS', '', '', '', '', '']
eventList = getEventList(toDo)
result1 /= getProbability(task3_model, eventList, 0, toDo, domain_list)
print(result1)

#p2 = P(Disability=' Severe ' | PatientAge='65+' , MRIScanResult=' Ischemic Stroke ')
toDo = ['65+', '', 'IS', '', '', '', 'Sev']
eventList = getEventList(toDo)
result2 = getProbability(task3_model, eventList, 0, toDo, domain_list)
toDo = ['65+', '', 'IS', '', '', '', '']
eventList = getEventList(toDo)
result2 /= getProbability(task3_model, eventList, 0, toDo, domain_list)
print(result2)

#p3 = P(StrokeType='Stroke Mimic' | PatientAge='65+' , CTScanResult='Hemmorraghic Stroke' , MRIScanResult='Ischemic Stroke')
eventList = [4, 5, 6]
toDo = ['65+', 'HS', 'IS', 'SM', '', '', '']
result3 = getProbability(task3_model, eventList, 0, toDo, domain_list)
eventList = [3, 4, 5, 6]
toDo = ['65+', 'HS', 'IS', '', '', '', '']
result3 /= getProbability(task3_model, eventList, 0, toDo, domain_list)
print(result3)

#p4 = P(Mortality='False' | PatientAge='0-30', Anticoagulants=’Used’, StrokeType='Stroke Mimic')
toDo = ['0-30', '', '', 'SM', 'U', 'F', '']
eventList = getEventList(toDo)
result4 = getProbability(task3_model, eventList, 0, toDo, domain_list)
toDo = ['0-30', '', '', 'SM', 'U', '', '']
eventList = getEventList(toDo)
result4 /= getProbability(task3_model, eventList, 0, toDo, domain_list)
print(result4)

#p5 = P(PatientAge='0-30', CTScanResult='Ischemic Stroke', MRIScanResult=' 'Hemmorraghic Stroke' , Anticoagulants=’Used’, StrokeType='Stroke Mimic' , Disability=' Severe' , Mortality ='False' )
toDo = ['0-30', '', '', 'SM', 'U', 'F', '']
result5 = task3_model.probability(['0-30','IS','HS','SM','U','F','Sev'])
print(result5)
