import numpy as np
import sys
import time
import random
import copy
import warnings
import copy

warnings.filterwarnings("ignore")

inf = float('inf')

# hyper paramters
ps_iter = 10
alpha = 3
psize = 30  # Population size
ubtrial = 50  # Maximum trials for generating initial solutions
pls = 0.5  # Probability of carrying out local search (mutation)


class CARPsolver():
    def __init__(self):
        self.vertices_number = 0
        self.depot_vertex = 0
        self.required_edges = 0
        self.nonrequired_edges = 0
        self.vehicles_number = 0
        self.vehicle_capacity = 0
        self.total_cost = 0
        self.edges_number = 0
        self.demand_list = []
        self.total_demand = 0

    def initCarp(self, filepath: str):
        f = open(filepath, 'r')
        name = f.readline()  # read name
        self.vertices_number = int(f.readline().strip().split(' ')[-1])
        self.depot_vertex = int(f.readline().strip().split(' ')[-1])
        self.required_edges = int(f.readline().strip().split(' ')[-1])
        self.nonrequired_edges = int(f.readline().strip().split(' ')[-1])
        self.edges_number = self.required_edges + self.nonrequired_edges
        self.vehicles_number = int(f.readline().strip().split(' ')[-1])
        self.vehicle_capacity = int(f.readline().strip().split(' ')[-1])
        self.total_cost = int(f.readline().strip().split(' ')[-1])
        self.cost_graph = np.full((self.vertices_number + 1, self.vertices_number + 1), inf)  # actual cost on each edge
        self.demand_graph = np.zeros((self.vertices_number + 1, self.vertices_number + 1))

        title = f.readline()  # read titles

        for i in range(self.edges_number):
            fractions = f.readline().strip().split()
            v1 = int(fractions[0])
            v2 = int(fractions[1])
            cost = int(fractions[2])
            demand = int(fractions[3])
            self.cost_graph[v1][v2] = cost
            self.cost_graph[v2][v1] = cost
            self.demand_graph[v1][v2] = demand
            self.demand_graph[v2][v1] = demand
            if demand != 0:
                self.demand_list.append((v1, v2))
                self.total_demand += demand

        f.close()

    def floyd(self):
        self.sp_graph = self.cost_graph.copy()  # shortest path

        for i in range(1, self.vertices_number + 1):
            self.sp_graph[i][i] = 0

        for k in range(1, self.vertices_number + 1):
            for i in range(1, self.vertices_number + 1):
                for j in range(1, self.vertices_number + 1):
                    if self.sp_graph[i][j] > self.sp_graph[i][k] + self.sp_graph[k][j]:
                        self.sp_graph[i][j] = self.sp_graph[i][k] + self.sp_graph[k][j]

    def getNearList(self, vh, demandList):
        nearList = []
        td = 0
        ned = 0
        rhs = self.total_cost / self.required_edges
        for vi, vj in demandList:
            minSP = min(self.cost_graph[vh][vj], self.cost_graph[vh][vi])
            if minSP <= rhs:
                nearList.append((vi, vj))
                td += self.demand_graph[vi][vj]
                ned += 1
        return nearList, td, ned

    def efficiencyRule(self, vh, edges, R):
        candidates = []
        temptcost = 0
        temptsp = 0
        temptDemand = 0
        eff = 0
        if R:
            for ui, vi in R:
                temptDemand += self.demand_graph[ui][vi]
                temptcost += self.cost_graph[ui][vi]
            for i in range(len(R) - 1):
                temptsp += self.sp_graph[R[i][1]][R[i + 1][0]]
            eff = temptDemand / (
                    self.sp_graph[self.depot_vertex][R[0][0]] + temptcost + temptsp + self.sp_graph[R[-1][1]][
                self.depot_vertex])
        for vi, vj in edges:
            lhs1 = self.demand_graph[vi][vj] / (self.sp_graph[vh][vi] + self.cost_graph[vi][vj] +
                                                self.sp_graph[vj][self.depot_vertex] - self.sp_graph[vh][
                                                    self.depot_vertex])
            lhs2 = self.demand_graph[vi][vj] / (self.sp_graph[vh][vj] + self.cost_graph[vi][vj] +
                                                self.sp_graph[vi][self.depot_vertex] - self.sp_graph[vh][
                                                    self.depot_vertex])
            if lhs1 >= eff or lhs2 >= eff:
                candidates.append((vi, vj))
        return candidates

    def getCandidateList(self, requiredEdges, rvc, vh):
        candidateList = []
        minDist = inf
        for e in requiredEdges:
            if self.demand_graph[e[0]][e[1]] <= rvc:
                if self.sp_graph[e[0]][vh] < minDist:
                    candidateList = [e]
                    minDist = self.sp_graph[e[0]][vh]
                elif self.sp_graph[e[1]][vh] < minDist:
                    candidateList = [e]
                    minDist = self.sp_graph[e[1]][vh]
                elif self.sp_graph[e[0]][vh] == minDist or self.sp_graph[e[1]][vh] == minDist:
                    candidateList.append(e)
        return candidateList

    def pathScanning(self):
        bestsol = []
        bestDist = inf
        for it in range(ps_iter):
            sol = []
            R = []
            rvc = self.vehicle_capacity
            vh = self.depot_vertex
            rule = False
            demandList = copy.deepcopy(self.demand_list)
            while demandList:
                nearList, neartd, nearned = self.getNearList(vh, demandList)
                if len(nearList) != 0 and rvc <= alpha * neartd / nearned:
                    rule = True
                elif len(nearList) == 0 and rvc <= alpha * self.total_demand / self.required_edges:
                    rule = True
                if rule:
                    candidates = self.getCandidateList(demandList, rvc, vh)
                    candidates = self.efficiencyRule(vh, candidates, R)
                else:
                    candidates = self.getCandidateList(demandList, rvc, vh)
                if len(candidates) == 0:
                    sol.append(R)
                    R = []
                    rvc = self.vehicle_capacity
                    vh = self.depot_vertex
                    rule = False
                else:
                    chosenEdge = random.choice(candidates)
                    demandList.remove(chosenEdge)
                    vi = chosenEdge[0]
                    vj = chosenEdge[1]
                    if self.sp_graph[vh][vi] > self.sp_graph[vh][vj]:
                        vi = chosenEdge[1]
                        vj = chosenEdge[0]
                    rvc = rvc - self.demand_graph[vi][vj]
                    vh = vj
                    R.append((vi, vj))
            sol.append(R)
            sumDist = self.getSolCost(sol)
            if sumDist < bestDist:
                bestsol = sol
                bestDist = sumDist
        return bestsol, bestDist

    def getSolCost(self, sol):
        sumDist = 0
        for R in sol:
            sumDist += self.sp_graph[self.depot_vertex][R[0][0]]
            sumDist += self.sp_graph[self.depot_vertex][R[-1][1]]
            for i in range(len(R) - 1):
                sumDist += self.sp_graph[R[i][1]][R[i + 1][0]]
        sumDist += self.total_cost
        return sumDist

    def solve(self):
        self.floyd()
        # sol, dist = self.pathScanning()
        # self.localSearch(sol)
        # self.formatOutput(sol, dist)

        sol = self.EvolutionaryAlgorithm()
        dist = self.getSolCost(sol)
        self.formatOutput(sol, dist)
        # debug
        # original = [[(2, 3)], [(3, 4)], [(5, 6)], [(6, 7), (7, 8)]]
        # son = self.combine([[(2, 3), (3, 4)], [(5, 6), (6, 7), (7, 8)]], 1, [(5, 6)], [(2, 3)], [(6, 7), (7, 8)])
        # self.check(son)

    def check(self, sol):
        if sol:
            demandList = copy.deepcopy(self.demand_list)
            for R in sol:
                if self.getRouteDemand(R) > self.vehicle_capacity:
                    print(sol)
                    raise Exception("larger than capacity")
                for task in R:
                    if task in demandList:
                        demandList.remove(task)
                    elif (task[1], task[0]) in demandList:
                        demandList.remove((task[1], task[0]))
                    else:
                        print("sol", sol)
                        raise Exception("edge not in demandlist")
            if demandList:
                print("sol, ", sol)
                raise Exception("there're edges not served")

    def EvolutionaryAlgorithm(self):
        psize = 30  # Population size
        ubtrial = 50  # Maximum trials for generating initial solutions
        pls = 0.5  # Probability of carrying out local search (mutation)
        pop = self.initialization(psize, ubtrial)
        pop.sort(key=self.getSolCost)
        psize = len(pop)
        opsize = 2 * psize  # No. of offspring generated in each generation, original is 6
        evolve_rate = 1.5
        select_rate = 0.3
        count = 0
        while time.time() - start < termination - 1:
            popt = copy.deepcopy(pop)
            for _ in range(0, opsize, 2):
                father_idx = random.randint(0, psize - 1)
                mother_idx = random.randint(0, psize - 1)
                while father_idx == mother_idx:
                    mother_idx = random.randint(0, psize - 1)
                son1, son2 = self.SBX(pop[father_idx], pop[mother_idx])
                # count += 1
                # print(count)
                # self.check(son1)
                # self.check(son2)
                if random.random() < pls:
                    son1 = self.localSearch(son1)
                    # self.check(son1)
                if random.random() < pls:
                    son2 = self.localSearch(son2)
                    # self.check(son2)
                if son1 and son1 not in popt:
                    popt.append(son1)
                if son2 and son2 not in popt:
                    popt.append(son2)
            popt.sort(key = self.getSolCost)
            pop = popt[0:int(psize*select_rate)]
            while len(pop) < psize:
                choice = int(random.random() ** evolve_rate * (len(popt) - len(pop))) + len(pop)
                if popt[choice] not in pop:
                    pop.append(popt[choice])
        return pop[0]


    def localSearch(self, son):
        # new_son = self.singleInsertion(son)
        # new_son2 = self.doubleInsertion(new_son)
        # new_son3 = self.swap(new_son2)
        # new_son4 = self.two_opt1(new_son3)
        # new_son5 = self.two_opt2(new_son4)
        weight = [0.35, 0.5, 0.7, 0.85]
        choice = random.random()
        if choice < weight[0]:
            new_son = self.singleInsertion(son)
            # print("singleInsertion, new_son, ", new_son)
        elif choice < weight[1]:
            new_son = self.doubleInsertion(son)
            # print("doubleInsertion, new_son, ", new_son)
        elif choice < weight[2]:
            new_son = self.swap(son)
            # print("swap, new_son, ", new_son)
        elif choice < weight[3]:
            new_son = self.two_opt1(son)
            # print("two_opt1, new_son ", new_son)
        else:
            new_son = self.two_opt2(son)
            # print("two_opt2, new_son ", new_son)
        if new_son:
            self.flip(new_son)
            # print("flip son, ", new_son)
        return new_son



    def singleInsertion(self, old_solution):
        new_solution = copy.deepcopy(old_solution)
        R_idx1 = random.randint(0, len(new_solution) - 1)
        R1 = new_solution[R_idx1]
        task_idx = random.randint(0, len(R1) - 1)
        task = R1[task_idx]
        demand = self.demand_graph[task[0]][task[1]]
        R1.pop(task_idx)
        R2_candidates = []
        for route_temp in new_solution:
            if self.vehicle_capacity - self.getRouteDemand(route_temp) >= demand:
                R2_candidates.append(route_temp)
        R2 = random.choice(R2_candidates)
        self.insertEdge(task, R2)
        # TODO: maybe insert to a random position is better
        self.removeEmptyRoute(new_solution)
        return new_solution


    def doubleInsertion(self, old_solution):
        new_solution = copy.deepcopy(old_solution)
        count = 0
        R_idx = 0
        while count <= len(new_solution):
            R_idx = random.randint(0, len(new_solution) - 1)
            count += 1
            if len(new_solution[R_idx]) >= 2:
                break
        if count > len(new_solution):
            return None
        R = new_solution[R_idx]
        task_idx = random.randint(0, len(R) - 2)
        task1 = R[task_idx]
        task2 = R[task_idx + 1]
        tasks_demand = self.demand_graph[task1[0]][task1[1]] + self.demand_graph[task2[0]][task2[1]]
        R.remove(task1)
        R.remove(task2)
        R2_candidates = []
        for route_temp in new_solution:
            if self.vehicle_capacity - self.getRouteDemand(route_temp) >= tasks_demand:
                R2_candidates.append(route_temp)
        R2 = random.choice(R2_candidates)
        # TODO: find the best place to insert these two tasks
        insert_idx = random.randint(0, len(R2))
        R2.insert(insert_idx, task1)
        R2.insert(insert_idx + 1, task2)
        self.removeEmptyRoute(new_solution)
        return new_solution

    def swap(self, old_solution):
        new_solution = copy.deepcopy(old_solution)
        r1 = random.randint(0, len(new_solution) - 1)
        r2 = random.randint(0, len(new_solution) - 1)
        R1 = new_solution[r1]
        R2 = new_solution[r2]
        demand1 = self.getRouteDemand(R1)
        demand2 = self.getRouteDemand(R2)
        count = 0
        while count <= len(R1) + len(R2):
            count += 1
            task1_idx = random.randint(0, len(R1) - 1)
            task2_idx = random.randint(0, len(R2) - 1)
            task1 = R1[task1_idx]
            task2 = R2[task2_idx]
            if task1 == task2:
                continue
            task1_demand = self.demand_graph[task1[0]][task1[1]]
            task2_demand = self.demand_graph[task2[0]][task2[1]]
            if demand1 - task1_demand + task2_demand <= self.vehicle_capacity and \
                demand2 - task2_demand + task1_demand <= self.vehicle_capacity:
                R1.remove(task1)
                self.insertEdge(task2, R1)
                R2.remove(task2)
                self.insertEdge(task1, R2)
                break
                # TODO: maybe insert to a random place is better
                # R1 = R1[:task1_idx] + [task2] + R1[task1_idx + 1:]
                # R2 = R2[:task2_idx] + [task1] + R2[task2_idx + 1:]
        if count > len(R1) + len(R2):
            return None
        return new_solution

    def two_opt1(self, old_solution):
        new_solution = copy.deepcopy(old_solution)
        R_idx = random.randint(0, len(new_solution) - 1)
        R = new_solution[R_idx]
        new_solution.pop(R_idx)
        sub1 = random.randint(0, len(R) - 1) # inclusive
        sub2 = random.randint(0, len(R) - 1) # inclusive
        if sub1 > sub2:
            sub1, sub2 = sub2, sub1
        subroute = []
        for i in range(sub2, sub1 - 1, -1):
            task = R[i]
            subroute.append((task[1], task[0]))
        R = R[:sub1] + subroute + R[sub2+1:]
        new_solution.append(R)
        return new_solution

    def two_opt2(self, old_solution):
        new_solution = copy.deepcopy(old_solution)
        R_idx1 = random.randint(0, len(new_solution) - 1)
        R_idx2 = random.randint(0, len(new_solution) - 1)
        while R_idx2 == R_idx1:
            R_idx2 = random.randint(0, len(new_solution) - 1)
        R1 = new_solution[R_idx1]
        R2 = new_solution[R_idx2]
        ntrial = 0
        while ntrial <= len(R1) + len(R2):
            ntrial += 1
            sub1 = random.randint(0, len(R1) - 1)
            sub2 = random.randint(0, len(R2) - 1)
            seg11 = R1[:sub1]
            seg12 = R1[sub1:]
            seg21 = R2[:sub2]
            seg22 = R2[sub2:]
            plan1_cost = inf
            plan2_cost = inf
            feasible = False
            if self.getRouteDemand(seg11 + seg22) <= self.vehicle_capacity and self.getRouteDemand(seg12 + seg21) <= self.vehicle_capacity:
                new_R1, R1_cost = self.combineSegs(seg11, seg22)
                new_R2, R2_cost = self.combineSegs(seg12, seg21)
                plan1_cost = R1_cost + R2_cost
                feasible = True
            if self.getRouteDemand(seg11 + seg21) <= self.vehicle_capacity and self.getRouteDemand(seg12 + seg22) <= self.vehicle_capacity:
                new_R3, R3_cost = self.combineSegs(seg11, seg21)
                new_R4, R4_cost = self.combineSegs(seg12, seg22)
                plan2_cost = R3_cost + R4_cost
                feasible = True
            if feasible:
                new_solution.remove(R1)
                new_solution.remove(R2)
                if plan1_cost < plan2_cost:
                    if new_R1:
                        new_solution.append(new_R1)
                    if new_R2:
                        new_solution.append(new_R2)
                    return new_solution
                else:
                    if new_R3:
                        new_solution.append(new_R3)
                    if new_R4:
                        new_solution.append(new_R4)
                    return new_solution
        return None


    def flip(self, solution):
        for R in solution:
            if len(R) == 1:
                continue
            for j in range(len(R)):
                if j == 0:
                    prev = self.depot_vertex
                    next = R[j + 1][0]
                elif j == len(R) - 1:
                    prev = R[j - 1][1]
                    next = self.depot_vertex
                else:
                    prev = R[j - 1][1]
                    next = R[j + 1][0]
                task = R[j]
                if self.sp_graph[prev][task[1]] + self.sp_graph[task[0]][next] < \
                        self.sp_graph[prev][task[0]] + self.sp_graph[task[1]][next]:
                    R[j] = (task[1], task[0])


    def SBX(self, father, mother):
        father_R_idx = random.randint(0, len(father) - 1)
        mother_R_idx = random.randint(0, len(mother) - 1)
        father_R = father[father_R_idx]  # father的某一条路径
        mother_R = mother[mother_R_idx]  # mother的某一条路径
        father_seg_idx = random.randint(0, len(father_R) - 1)
        mother_seg_idx = random.randint(0, len(mother_R) - 1)
        father_seg1 = father_R[0:father_seg_idx]
        father_seg2 = father_R[father_seg_idx:]
        mother_seg1 = mother_R[0:mother_seg_idx]
        mother_seg2 = mother_R[mother_seg_idx:]
        # father_seg1 & mother_seg2, father_seg2 & mother_seg1
        son1 = self.combine(father, father_R_idx, father_seg1, mother_seg2, father_seg2)
        son2 = self.combine(mother, mother_R_idx, mother_seg1, father_seg2, mother_seg2)
        return son1, son2


    def combine(self, original, R_idx, seg1, new_seg2, old_seg2) -> list:
        son = copy.deepcopy(original)
        del son[R_idx]
        # remove duplicate task, because seg1 is from original, duplicated task must happen in seg2
        # we only need to remove duplicated task in seg2
        seg = copy.deepcopy(seg1)
        son.append(seg)
        duplicated_edges = []
        for task in new_seg2:
            if task not in old_seg2 and (task[1], task[0]) not in old_seg2:
                duplicated_edges.append(task)
        for task in duplicated_edges:
            for R in son:
                if task in R:
                    R.remove(task)
                elif (task[1], task[0]) in R:
                    R.remove((task[1], task[0]))

        # recover the missed
        missed_edges = copy.deepcopy(new_seg2)
        for task in old_seg2:
            if task not in new_seg2 and (task[1], task[0]) not in new_seg2:
                missed_edges.append(task)
        for task in missed_edges:
            newRouteRVC = self.vehicle_capacity - self.getRouteDemand(seg)
            if newRouteRVC >= self.demand_graph[task[0]][task[1]]: #插入seg
                self.insertEdge(task, seg)
            else: # 插入已有的路径
                offset = random.randint(0, len(son) - 1)
                feasible = False
                for i in range(len(son)):
                    R_idx = (i + offset) % len(son)
                    R = son[R_idx]
                    rvc = self.vehicle_capacity - self.getRouteDemand(R)
                    if rvc < self.demand_graph[task[0]][task[1]]:
                        continue
                    feasible = True
                    self.insertEdge(task, R)
                    break
                if not feasible: # 如果无法插入已有的路径，那么就创建新的路径
                    son.append([task])
        self.removeEmptyRoute(son)
        return son

    def removeEmptyRoute(self, sol):
        while [] in sol:
            sol.remove([])

    def getRouteDemand(self, R):
        demand = 0
        for edge in R:
            demand += self.demand_graph[edge[0]][edge[1]]
        return demand

    def combineSegs(self, seg1, seg2):
        tempt1 = seg1 + seg2
        tempt1_cost = 0
        if tempt1:
            tempt1_cost = self.getRouteAdditionalCost(tempt1)
        tempt2 = seg2 + seg1
        tempt2_cost = 0
        if tempt2:
            tempt2_cost = self.getRouteAdditionalCost(tempt2)
        if tempt1_cost < tempt2_cost:
            return tempt1, tempt1_cost
        return tempt2, tempt2_cost

    def getRouteAdditionalCost(self, R):
        cost = 0
        for i in range(len(R) - 1):
            cost += self.sp_graph[R[i][1]][R[i + 1][0]]
        cost += self.sp_graph[self.depot_vertex][R[0][0]] + self.sp_graph[self.depot_vertex][R[-1][1]]
        return cost


    def insertEdge(self, task, R):
        if not R:
            R.append(task)
            return
        minCost = inf
        position = 0
        minTask = task
        for j in range(len(R) + 1):
            if j == 0:
                additionalCost, tempTask = self.getInsertEdgeCost(task, self.depot_vertex, R[j][0])
            elif j == len(R):
                additionalCost, tempTask = self.getInsertEdgeCost(task, R[j - 1][1], self.depot_vertex)
            else:
                additionalCost, tempTask = self.getInsertEdgeCost(task, R[j - 1][1], R[j][0])
            if additionalCost < minCost:
                minCost = additionalCost
                position = j
                minTask = tempTask
        R.insert(position, minTask)


    def getInsertEdgeCost(self, task, prev_point, next_point):
        if self.sp_graph[prev_point][task[0]] + self.sp_graph[task[1]][next_point] < \
                self.sp_graph[prev_point][task[1]] + self.sp_graph[task[0]][next_point]:
            minCost = self.sp_graph[prev_point][task[0]] + self.sp_graph[task[1]][next_point]
            minTask = (task[0], task[1])
        else:
            minCost = self.sp_graph[prev_point][task[1]] + self.sp_graph[task[0]][next_point]
            minTask = (task[1], task[0])
        return minCost, minTask


    def initialization(self, psize, ubtrial):
        pop = []
        ntrial = 0
        while len(pop) < psize and ntrial < ubtrial:
            ntrial += 1
            sol, _ = self.pathScanning()
            if sol not in pop:
                pop.append(sol)
        return pop

    def formatOutput(self, sol, dist):
        s = "s "
        for r in sol:
            s = s + "0,"
            for d in r:
                s += "(%d,%d)," % (d[0], d[1])
            s = s + "0,"
        s = s[:-1]
        q = "q " + str(int(dist))
        print(s)
        print(q)


# start = time.time()
# filepath = sys.argv[1]
# termination = sys.argv[3]
# random_seed = sys.argv[5]
filepath = 'CARP_samples/egl-s1-A.dat'
termination = 60
random_seed = 1
random.seed(random_seed)
carp_solver = CARPsolver()
carp_solver.initCarp(filepath)
start = time.time()
carp_solver.solve()
