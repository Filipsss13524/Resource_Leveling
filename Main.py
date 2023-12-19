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
            if (element[0], a) in c.relation_list and a not in critical_path:
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
        print(los[0])
        if z[c.enum_task[los[0]]] != 0:
            zn = random.randint(1, z[c.enum_task[los[0]]])
            nroz[c.enum_data[los[0]]] = 0
            nroz[c.enum_data[los[0]], Tw[c.enum_task[los[0]]] + zn:Tw[c.enum_task[los[0]]] + data.loc[c.enum_data[los[0]], 'Time'] + zn] = 1
            nroz_m = f_move(nroz, los[0], Tw[c.enum_task[los[0]]] + data.loc[c.enum_data[los[0]], 'Time'] + zn)
            print(nroz_m)
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
    index = 0
    for pi in population:
        res = np.zeros((res_num, len(pi[0])))
        for i in range(len(pi[0])):
            for j in range(row):
                if pi[j][i] == 1:
                    for idx in range(res_num):
                        res[idx][i] += data.loc[j, f'R{idx + 1}']
                if pi[j][i] == 2:
                    for idx in range(res_num):
                        res[idx][i] += data.loc[j, f'R{idx + 1}'] / 2
        Resource.append(res)

    if goal_f == 1:
        for pf in range(len(population)):
            fit = np.zeros(res_num)
            for l in range(res_num):
                fit[l] = round(weight[l] * np.max(Resource[pf][l]) / sum(data.loc[:, f'R{l + 1}']), 3) * 10
            Fitness[index] = sum(fit) + len(Resource[pf][0]) - Tp[-1]
            index += 1

    elif goal_f == 2:
        for pf in range(len(population)):
            fit = np.zeros(res_num)
            for l in range(res_num):
                fit[l] = round(weight[l] * sum(np.abs(np.mean(Resource[pf][l]) - Resource[pf][l])), 3)
            Fitness[index] = sum(fit) + len(Resource[pf][0]) - Tp[-1]
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
        weight = val / sum(val) * 100
        while len(new_list) != size:
            licz = random.choices(list(fitness.keys()), weights=weight, k=15)
            if licz[0] not in new_list:
                new_list.append(licz[0])
                weight[licz[0]] = 0.5
        for idx, i in enumerate(new_list):
            new_population.append(population[i])
            new_resources.append(resource[i])
            new_fitness[idx] = fitness[i]
        print(len(new_population))

    elif mode == 2:  # rankingowa
        new_list = sorted(fitness, key=fitness.get)[0:size]
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
    if len(rozf[0]) == len(rozg[0]):
        for i in lis:
            l = c.enum_data[i]
            if np.all(rozf[l] == rozg[l]):
                continue
            else:
                where_rozf = np.argwhere(rozf[l] != 0)
                where_rozg = np.argwhere(rozg[l] != 0)
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

    return population


def mutation(population, resources, mutation_list, limit_block, optim_resource = 1):
    for i in mutation_list:
        act_pop = population[i].copy()
        max_w = np.abs(max(resources[i][optim_resource - 1]) - np.mean(resources[i][optim_resource - 1]))
        min_w = np.abs(min(resources[i][optim_resource - 1]) - np.mean(resources[i][optim_resource - 1]))
        lis = [c.enum_data[x] for x in c.enum_task.keys() if x not in critical_path]  # nie list lecz sądziedzi
        lis2 = [x for x in c.enum_task.keys() if x not in critical_path]
        k = list(c.enum_data.keys())
        v = list(c.enum_data.values())
        go_min = 0
        maxxx = np.linspace(1,int(L_iter/10),int(L_iter/10))
        if Iter / 10 in maxxx:
            max_w += 2
        if max_w > min_w:
            print("max")
            print(act_pop)
            index_list = np.where(resources[i][0] == max(resources[i][0]))[0]
            ile_dz = 0
            if limit_block[1] == 1:
                for x in index_list:
                    list_wynik = []
                    for licz, il in enumerate(act_pop[:, x]):
                        if licz in lis and il == 1:
                            cz = lis2[lis.index(licz)]
                            rel_mat = [licz]
                            for rel in c.enum_data:
                                if (rel, cz) in c.relation_list or (cz, rel) in c.relation_list:
                                    num_rel = c.enum_data[rel]
                                    rel_mat.append(num_rel)
                            where_roz = np.argwhere(act_pop[licz] != 0)
                            if x != 0 and np.all(act_pop[rel_mat, x - 1] == 0):
                                list_wynik.append((1, licz, data.loc[licz, f'R{optim_resource}']))  # left bliski
                            if x != len(population[i][0]) - 1 and np.all(act_pop[rel_mat, x + 1] == 0):
                                list_wynik.append((2, licz, data.loc[licz, f'R{optim_resource}']))  # right bliski

                    if list_wynik:
                        num = random.randint(0, (len(list_wynik)) - 1)
                        w = list_wynik[num]
                        if w[0] == 1:
                            act_pop[w[1], x-1:x+1] = 2
                            ile_dz += 1
                        if w[0] == 2:
                            act_pop[w[1], x:x+2] = 2
                            ile_dz += 1

            if ile_dz == 0:  # wydłużenie czasu lub uskok przed lokalem
                go_min = 1
                if limit_block[2] == 1:  # dodanie dnia
                    act_new = np.zeros((len(act_pop), len(act_pop[0]) + 1))
                    for licz, m in enumerate(range(len(act_pop))):
                        w = list(act_pop[m])
                        w.insert(index_list[0],0)
                        if licz in lis and act_pop[m,index_list[0]] != 0:
                            w[index_list[0]] = act_pop[m,index_list[0]]
                            w[index_list[0] + 1] = 0

                        act_new[m] = w
                        population.append(act_new)
            else:
                print(act_pop)
                population.append(act_pop)

        if min_w > max_w or go_min == 1:
            print("min")
            # print(act_pop)
            index_list = list(np.where(resources[i][optim_resource - 1] == min(resources[i][optim_resource - 1]))[0])
            for x in reversed(index_list):
                idx_l = -1
                idx_p = -1
                print(act_pop)
                for licz, il in enumerate(act_pop[:, x]):
                    if il == 1:
                        cz = k[v.index(licz)]
                        rel_mat = [licz]
                        for rel in c.enum_data:
                            if (rel, cz) in c.relation_list or (cz, rel) in c.relation_list:
                                num_rel = c.enum_data[rel]
                                rel_mat.append(num_rel)
                for y1 in reversed(range(x)):
                    if sum(act_pop)[y1] > 1:
                        idx_l = y1
                        break
                if x + 1 != len(population[i][0]):
                    for y2 in range(x + 1, len(population[i][0])):
                        if sum(act_pop)[y2] > 1:
                            idx_p = y2
                            break

                list_wynik = []
                if idx_l != -1:
                    for licz, il in enumerate(act_pop[:, idx_l]):
                        if licz in lis and il == 1 and licz not in rel_mat:
                            list_wynik.append((idx_l, licz, data.loc[licz, f'R{optim_resource}']))
                        if licz in lis and il == 2 and licz not in rel_mat:
                            list_wynik.append((idx_l, licz, data.loc[licz, f'R{optim_resource}']/2))
                if idx_p != -1:
                    for licz, ip in enumerate(act_pop[:, idx_p]):
                        if licz in lis and ip == 1 and licz not in rel_mat:
                            list_wynik.append((idx_p, licz, data.loc[licz, f'R{optim_resource}']))
                        if licz in lis and ip == 2 and licz not in rel_mat:
                            list_wynik.append((idx_p, licz, data.loc[licz, f'R{optim_resource}']/2))


                if len(list_wynik) == 0:
                    continue
                # if go_min == 0:
                num = random.randint(0, (len(list_wynik)) - 1)
                maximum = list_wynik[num][2]
                z = list_wynik[num][1]
                idx = list_wynik[num][0]

                if limit_block[0] == 1:  # Rozdzielanie zasobów
                    if act_pop[z,idx] == 2 and act_pop[z, x] == 2:
                        act_pop[z, x] = 1
                        act_pop[z, idx] = 0
                    else:
                        act_pop[z, x], act_pop[z, idx] = act_pop[z, idx], act_pop[z, x]

                else:  # Alokacja zasobów
                    if act_pop[z, x] == 0:
                        if x > idx:  # przesunięcie w prawo
                            L_z = list(act_pop[z].copy())
                            l_elem = x - idx
                            for ll in range(l_elem):
                                L_z.pop(-1)
                                L_z.insert(0, 0)
                            act_pop[z] = L_z

                        else:  # przesunięcie w lewo
                            L_z = list(act_pop[z].copy())
                            l_elem = idx - x
                            for ll in range(l_elem):
                                L_z.pop(0)
                                L_z.append(0)
                            act_pop[z] = L_z

            # print(act_pop)
            population.append(act_pop)

    return population


def plot_resource(solution, resource_nb=1, start = 1, xl=0, yl=0):
    new_solution = []
    print(solution)
    for idx, i in enumerate(solution):
        w1 = np.where(i == 1, data.loc[idx, f'R{resource_nb}'], 0)
        w2 = np.where(i == 2, data.loc[idx, f'R{resource_nb}'] / 2, 0)
        new_solution.append(w1 + w2)

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
    if resource_nb == 1 and start == 1:
        plt.savefig("wynik1_s.png")
        plt.show()
    if resource_nb == 2 and start == 1:
        plt.savefig("wynik2_s.png")
        plt.show()
    if resource_nb == 3 and start == 1:
        plt.savefig("wynik3_s.png")
        plt.show()
    if resource_nb == 4 and start == 1:
        plt.savefig("wynik4_s.png")
        plt.show()
    if resource_nb == 5 and start == 1:
        plt.savefig("wynik5_s.png")
        plt.show()
    if resource_nb == 1 and start == 0:
        plt.savefig("wynik1.png")
        plt.show()
    if resource_nb == 2 and start == 0:
        plt.savefig("wynik2.png")
        plt.show()
    if resource_nb == 3 and start == 0:
        plt.savefig("wynik3.png")
        plt.show()
    if resource_nb == 4 and start == 0:
        plt.savefig("wynik4.png")
        plt.show()
    if resource_nb == 5 and start == 0:
        plt.savefig("wynik5.png")
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
    data = pd.read_excel("Scenariusz 2.xlsx")  # Tabela z bazą danych
    size = 50  # Rozmiar populacji
    L_iter = 1000  # Liczba iteracji algorytmu
    goal_function = 2  # Funkcja celu 1. maksymlany zasób, 2. Suma odchyleń od średniej
    resource_number = 2  # Liczba zasobów dla projektu
    weight_resource = [1, 1]  # Wagi dla poszczególnych zasobów
    selection_mode = 1  # Obcja wyboru selekcji 1. Koło ruletki, 2. Selekcja rankingowa
    limit_block = [1, 1, 0]  # Blokady dla opodwiednich mutacji w algorytmie 1 - włączone 0 - wyłączone, 1. Rozdzielanie zadań, 2. Dzielenie zasobów na pół, 3. Wydłużanie czasu

    # CPM
    wynik = []
    c = CPM.CPM(data)
    Tw, Tp, z = c.define_deadlines()
    critical_path = c.get_critical_path(z)
    # Algorytm
    Iter = 0
    roz = first_result()
    new_pop = create_population(first_soluction=roz, size=size)
    while Iter < L_iter:
        fit, res = fitness(population=new_pop, goal_f=goal_function, res_num=resource_number, weight=weight_resource)
        print(fit)
        w = min(fit, key=fit.get)
        print(w)
        wynik.append(new_pop[w])
        new_pop, new_res, new_fit = selection(population=new_pop, fitness=fit, resource=res, mode=selection_mode,size=30)
        # Losowanie 2 elementów do krzyżwoanie jesli będziemy chcieli zmieniać konkretne należy to zmienić
        l1 = random.choice(list(new_fit.keys()))
        l2 = random.choice(list(new_fit.keys()))
        while l1 == l2:
            l2 = random.choice(list(new_fit.keys()))
        new_pop = cross(population=new_pop, f=l1, g=l2)
        m_licz = math.ceil(len(new_pop) * 0.1)
        m_list = []
        for i in range(m_licz):
            l = random.choice(list(new_fit.keys()))
            while l in m_list:
                l = random.choice(list(new_fit.keys()))
            m_list.append(l)
        optim = random.choices([1,2],weight_resource, k = 1) #wybór optymalizacji parametrycznej
        new_pop = mutation(population=new_pop, resources=new_res, mutation_list=m_list, limit_block=limit_block, optim_resource=optim[0])
        Iter += 1

    fit, res = fitness(population=new_pop, goal_f=goal_function, res_num=resource_number, weight=weight_resource)
    w = min(fit, key=fit.get)
    wynik.append(new_pop[w])
    print("wynik")
    print(fit)
    print(w)
    plot_resource(roz, 1)
    print(roz)
    plot_resource(wynik[-1], 1)
    plot_resource(wynik[-1], 2)

