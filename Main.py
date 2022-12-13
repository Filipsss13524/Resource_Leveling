import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import CPM
import math
import plotly.express as px
import datetime


def first_result():
    """
    Funkcja tworząca pierwsze rozwiązanie na podstawie wyznaczonych najwcześniejszych terminów:
    return: Pierwsze rozwiążanie zakodowane binarnie [0,1]
    """
    roz = np.zeros((len(c.enum_data), Tw[-1]))
    for idx, d in enumerate(c.enum_data.keys()):
        roz[idx, Tw[c.enum_task[d]]:Tw[c.enum_task[d]] + data.loc[c.enum_data[d], 'Time']] = 1
    return roz


def f_move(nroz, name, point):
    """
    Funkcja przesuwająca wszystkie czynności kolidujące:
    param nroz: Rozwiązanie:
    param name: Nazwa czynności, której sąsiadów przesuwamy jak potrzeba:
    param point: Numer pierwszego wolnego miejsca:
    return: Nowe rozwiązniae
    """
    ns = point
    move_list = [[name, ns]]
    for element in move_list:
        for a in c.enum_data.keys():
            if (element[0], a) in c.relation_list:
                nroz[c.enum_data[a]] = 0
                nroz[c.enum_data[a], element[1]:element[1] + data.loc[c.enum_data[a], 'Time']] = 1
                move_list.append((a, element[1] + data.loc[c.enum_data[a], 'Time']))
    return nroz


def create_population(first_soluction, size):
    """
    Funkcja tworząca populację startową, losując czynność, która nie jest w ścieżce krytycznej oraz
    miejsce, gdzie może zostać ona przestawiona:
    param first_soluction: Pierwsze rozwiązanie:
    param size: Rozmiar populacji:
    return: Losowa populacja
    """
    population = [first_soluction]
    lis = [x for x in c.enum_task.keys() if x not in critical_path]
    for i in range(size - 1):
        nroz = first_soluction.copy()
        los = random.sample(lis, 1)
        if z[c.enum_task[los[0]]] != 0:
            zn = random.randint(1, z[c.enum_task[los[0]]])
            nroz[c.enum_data[los[0]]] = 0
            nroz[c.enum_data[los[0]],
            Tw[c.enum_task[los[0]]] + zn:Tw[c.enum_task[los[0]]] + data.loc[c.enum_data[los[0]], 'Time'] + zn] = 1
            nroz_m = f_move(nroz, los[0], Tw[c.enum_task[los[0]]] + data.loc[c.enum_data[los[0]], 'Time'] + zn)
            population.append(nroz_m)
        else:
            population.append(nroz)

    return population


def fitness(population, goal_f, res_num, weight):
    """
    Funkcja wyznaczająca ocenę dla każdego osobnika z populacji:
    param population: Populacja:
    param goal_f: Funkcja celu 1. maksymalna wartość, 2. odchylenie standardowe:
    param res_num: Liczba zasobów użytych w algorytmie:
    param size: Rozmiar populacji:
    param weight: Wagi dla każdego z zasobów:
    return: Ocena oraz macierz przystosowania
    """
    Resource = []
    Fitness = {}
    row = len(c.enum_data)
    col = Tw[-1]
    index = 0
    for pi in population:
        res = np.zeros((res_num, col))
        for i in range(col):
            for j in range(row):
                # jeśli będzie przerabiana funkcja celu to w tym miejscu 2,3 itd
                if pi[j][i] == 1:
                    for idx in range(res_num):
                        res[idx][i] += data.loc[j, f'R{idx+1}']
        Resource.append(res)

    if goal_f == 1:
        for pf in range(len(population)):
            fit = np.zeros(res_num)
            for l in range(res_num):
                fit[l] = round(weight[l] * np.max(Resource[pf][l]) / sum(data.loc[:, f'R{l+1}']), 3)
            Fitness[index] = sum(fit)
            index += 1

    elif goal_f == 2:
        for pf in range(len(population)):
            fit = np.zeros(res_num)
            for l in range(res_num):
                fit[l] = round(weight[l] * sum(np.abs(np.mean(Resource[pf][l]) - Resource[pf][l])), 3)
            Fitness[index] = sum(fit)
            index += 1
    else:
        print("Wybrana zła funkcja celu")

    return Fitness, Resource


def selection(population, fitness, resource, mode, size):
    """
    Funckja selekcji osobników do nowej populacji dwoma metodami: 1. koło ruletki, 2. selekcja rankingowa:
    param population: Populacja:
    param fitness: Funkcja celu:
    param resource: Macierz zasobowa wyznaczona na podtawie funkcji celu:
    param mode: Wybór metody selekcji:
    param size: Rozmiar nowej populacji po selekcji:
    return: new_population, new_resources, new_fitness
    """

    new_population = []
    new_fitness = {}
    new_resources = []
    if mode == 1:  # koło ruletki
        new_list = []
        val = 1 / (list(fitness.values()) / sum(fitness.values()))
        weight = val / sum(val)
        while len(new_list) != size:
            licz = random.choices(list(fitness.keys()), weights=weight, k=15)
            if licz[0] not in new_list:
                new_list.append(licz[0])
        for idx, i in enumerate(new_list):
            new_population.append(population[i])
            new_resources.append(resource[i])
            new_fitness[idx] = fitness[i]

    elif mode == 2:  # rankingowa
        new_list = sorted(fitness, key = fitness.get)[0:size]
        for idx, i in enumerate(new_list):
            new_population.append(population[i])
            new_resources.append(resource[i])
            new_fitness[idx] = fitness[i]
    else:
        print("Wybrano zły tryb")

    return new_population, new_resources, new_fitness


def cross(population, f, g):
    """
    Funkcja krzyżowania dwóch osobników:
    param population: Populacja:
    param f: Funckja 1:
    param g: Funckja 2:
    return: Nowa Populacja
    """
    lis = [x for x in c.enum_task.keys() if x not in c.get_critical_path(z)]
    rozf = population[f]
    rozg = population[g]
    for i in lis:
        l = c.enum_data[i]
        if np.all(rozf[l] == rozg[l]):
            continue
        else:
            where_rozf = np.argwhere(rozf[l] == 1)
            where_rozg = np.argwhere(rozg[l] == 1)
            where_start = [where_rozf[0][0] if where_rozf[0][0] < where_rozg[0][0] else where_rozg[0][0]]
            where_end = [where_rozf[-1][0] if where_rozf[-1][0] > where_rozg[-1][0] else where_rozg[-1][0]]
            rel_mat = []
            for rel in c.enum_data:
                if (rel, i) in c.relation_list or (i, rel) in c.relation_list:
                    num_rel = c.enum_data[rel]
                    rel_mat.append(num_rel)

            rozf_copy = rozf.copy()
            rozg_copy = rozg.copy()
            sum_rozf = np.sum(rozf[rel_mat, where_start[0]:where_end[0] + 1])
            sum_rozg = np.sum(rozg[rel_mat, where_start[0]:where_end[0] + 1])
            if sum_rozf == 0:
                rozf_copy[l] = rozg[l]
                population.append(rozf_copy)
            if sum_rozg == 0:
                rozg_copy[l] = rozf[l]
                population.append(rozg_copy)
            if sum_rozf != 0 and sum_rozg != 0 and (data.loc[l, 'Time'] < where_end[0] - where_start[0] - sum_rozg or data.loc[l, 'Time'] < where_end[0] - where_start[0] - sum_rozf):
                w_rozf = rozf_copy[rel_mat, where_start[0]:where_end[0] + 1]
                w_rozg = rozg_copy[rel_mat, where_start[0]:where_end[0] + 1]
                if (sum(w_rozf[:][0]) > 0 or sum(w_rozg[:][0]) > 0) and (sum(w_rozf[:][-1]) > 0 or sum(w_rozg[:][-1]) > 0):
                    nc = np.zeros(len(population[0][0]))
                    for idx,x,y in zip(np.linspace(where_start[0],where_end[0],len(w_rozf[0])),sum(w_rozf), sum(w_rozg)):
                        if x == 0 and y == 0 and sum(nc) < data.loc[l, 'Time']:
                            nc[int(idx)] = 1
                    rozg_copy[l] = nc[:]
                    rozf_copy[l] = nc[:]
                    population.append(rozg_copy)
                    population.append(rozf_copy)
                else:
                    continue

    return population


def mutation(population,resources,mutation_list):
    r = 1
    for i in mutation_list:
        print(resources[i])
        print(np.mean(resources[i][0]))
        act_pop = population[i]
        max_w = np.abs(max(resources[i][r-1]) - np.mean(resources[i][r-1]))
        min_w = np.abs(min(resources[i][r-1]) - np.mean(resources[i][r-1]))
        if min_w > max_w:
            index_list = np.where(resources[i][r-1] == min(resources[i][r-1]))[0]
            for x in reversed(index_list):
                idx_l = -1
                idx_p = -1
                for y1 in reversed(range(x)):
                    if sum(act_pop)[y1] > 1:  # index skąd bierzemy elementy
                        idx_l = y1
                        break
                if x+1 != len(population[0][0]):
                    for y2 in range(x + 1, len(population[0][0])):
                        if sum(act_pop)[y2] > 1:  # index skąd bierzemy elementy
                            idx_p = y2
                            break
                list_wynik = []
                lis = [c.enum_data[x] for x in c.enum_task.keys() if x not in critical_path]
                if idx_l != -1:
                    for licz, il in enumerate(act_pop[:,idx_l]):
                        if licz in lis and il == 1:
                            list_wynik.append((idx_l, licz, data.loc[licz, f'R{r}']))
                if idx_p != -1:
                    for licz, ip in enumerate(act_pop[:,idx_p]):
                        if licz not in lis and ip == 1:
                            list_wynik.append((idx_p, licz, data.loc[licz, f'R{r}']))
                maximum = 0
                z = 0
                idx = 0
                for m in list_wynik:
                    if m[2] > maximum:
                        maximum = m[2]
                        z = m[1]
                        idx = m[0]

                act_pop[z,idx] = 0
                act_pop[z,x] = 1
                population[i] = act_pop

        else:
            index_list = np.where(resources[i][0] == max(resources[i][0]))[0]
            print(index_list)
            print(population[i])

    return population


def plot_resource(solution, resource_nb=1, xl=0, yl=0):
    new_solution = []
    for idx, i in enumerate(solution):
        w = np.where(i == 1, data.loc[idx, f'R{resource_nb}'], 0)
        new_solution.append(w)

    fig, ax = plt.subplots()
    N = len(solution[0])
    ind = np.arange(1, N + 1)
    pop_p = np.zeros(N)
    for i in range(len(solution)):
        p = ax.bar(ind, new_solution[i], bottom=pop_p, label=data.loc[i, 'Task'])
        pop_p += new_solution[i]

    if xl != 0:
        ax.set_xlim(0, xl)
    if yl != 0:
        ax.set_ylim(0, yl)
    ax.set_title(f'Wykres zapotrzebowania dla zasobu R{resource_nb}')
    ax.set_ylabel("Zasób")
    ax.set_xlabel("Dzień")
    ax.legend()
    plt.show()


def plot_resource_mean(resource, resource_nb=1, xl=0, yl=0):
    fig, ax = plt.subplots()
    N = len(resource[0])
    ind = np.arange(1, N + 1)
    fit = [np.mean(resource[resource_nb - 1])] * N
    ax.bar(ind, resource[resource_nb - 1], label=f'R{resource_nb}')
    ax.plot(ind, fit, 'red', label="Średnia")

    if xl != 0:
        ax.set_xlim(0, xl)
    if yl != 0:
        ax.set_ylim(0, yl)
    ax.set_title(f'Wykres zapotrzebowania dla zasobu R{resource_nb}')
    ax.set_ylabel("Zasób")
    ax.set_xlabel("Dzień")
    ax.legend()
    plt.show()


def plot_Gantt():  # problem z przerwami i przekroczeniem czasu czy może może zosatc w formie macierzy katóra w miejscah jedynie wskazuje to co trzeba

    print("nie")


def plot_all_resources(resources, xl=0, yl=0):
    fig, ax = plt.subplots()
    N = len(resources[0])
    ind = np.arange(1, N + 1)
    pop_p = np.zeros(N)
    idx = 1
    for i in range(len(resources)):
        p = ax.bar(ind, resources[i], bottom=pop_p, label=f'R{idx}')
        pop_p += resources[i]
        idx += 1

    if xl != 0:
        ax.set_xlim(0, xl)
    if yl != 0:
        ax.set_ylim(0, yl)
    ax.set_title('Wykres zapotrzebowania dla zasobów')
    ax.set_ylabel("Zasób")
    ax.set_xlabel("Dzień")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Struktury danych
    data = pd.read_excel("Zestaw3.xlsx")  # Tabela z bazą danych
    size = 20  # Rozmiar populacji
    L_iter = 30  # Liczba iteracji algorytmu
    goal_function = 2  # Funkcja celu 1. maksymlany zasób, 2. Suma odchyleń od średniej
    resource_number = 2  # Liczba zasobów dla projektu
    weight_resource = [2, 1, 1, 1, 1]  # Wagi dla poszczególnych zasobów
    selection_mode = 1  # Obcja wyboru selekcji 1. Koło ruletki, 2. Selekcja rankingowa

    # CPM
    wynik = []
    c = CPM.CPM(data)
    Tw, Tp, z = c.define_deadlines()
    critical_path = c.get_critical_path(z)
    # Algorytm
    Iter = 0
    roz = first_result()
    new_pop = create_population(first_soluction=roz, size = size)
    while Iter < L_iter:
        fit, res = fitness(population=new_pop, goal_f=goal_function, res_num=resource_number, weight=weight_resource)
        w = min(fit, key=fit.get)
        wynik.append(new_pop[w])
        new_pop, new_res, new_fit = selection(population=new_pop,fitness=fit,resource=res, mode=selection_mode, size=15)
        # Losowanie 2 elementów do krzyżwoanie jesli będziemy chcieli zmieniać konkretne należy to zmienić
        l1 = random.choice(list(new_fit.keys()))
        l2 = random.choice(list(new_fit.keys()))
        while l1 == l2:
            l2 = random.choice(list(new_fit.keys()))
        new_pop = cross(population=new_pop, f=l1, g =l2)
        # losowane osobniki, które nie były poddane krzyżowaniu
        m_licz = math.ceil(len(new_pop)*0.1)
        m_list = []
        for i in range(m_licz):
            l = random.choice(list(new_fit.keys()))
            while l in m_list:
                l = random.choice(list(new_fit.keys()))
            m_list.append(l)
        new_pop = mutation(population=new_pop, resources=new_res, mutation_list= m_list)
        Iter += 1

    print(wynik)
    plot_resource(wynik[-1], 1)
    # plot_Gantt(roz)
    # pop = create_population(roz, 15)
    # fit, res = fitness(pop,1,2,15,[1,1,1,1,1])
    # l1,l2 = roulette(fit,2)
    # cross(pop,l1,l2)
    # mutation(pop, fit,res, 1)
    # print(pop[11])
    # print(res[11][0])
    # fig = plt.figure(figsize=(7,5))
    # ax = fig.add_axes([0,0,1,1])
    # ax.bar(np.linspace(0,13, num = 14),res[11][0])
    # plt.show()
    # for x,y,z in zip(pop,fit,res):
    #     print(x)
    #     print(y)
    #     print(z)
    # w = np.array([0,0,0,1,1,1,0])
    # print(np.argwhere(w==1))
    # print(roz[[1,2],2:7])
    # print(data.loc[1,'Time'])
    # w = np.array([0,1,1,1,0,0])
    # Wynik = np.where(w == 1,100,0)
# print(data.loc[1,f'R{1}'])
#     c2 = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]])
#
#     c1 = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])
#     pop = [c1,c2]
#     print(cross(pop,0,1))


