################################################################################################################
#                                                                                                              #
#                                           IMPORTING LIBS                                                     #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################


import numpy as np

import time

import pandas as pd

import math

import time

from threading import Thread




def intersect1d_searchsorted(A,B):

    idx = np.searchsorted(B,A)

    idx[idx==len(B)] = 0
    return A[B[idx] == A][0]


# Vamos construir uma árvore binária, que armazene nossa hierarquia
# Nó folha, não será representado. (Não é interessante para o HAI, já que não iremos comparar)

class Node:
    def __init__(self, key):
        self.parent = None

        self.left = None
        self.right = None
        self.value = key
        self.indexes = []
        self.size = None

### HAI
class HAI:


    def __init__(self,linkage,N) -> None:
        
        self.linkage = linkage
        self.N = N

    def build_hierarchy(self):
        nodes = (self.N-1 + self.N) #root

        root = Node(nodes-1)


        # Construir a árvore
        current_node = root

        # Irá armazenar os nós da árvore
        self.node_dict = {}

        # Adiciona a raiz ao dicionário
        self.node_dict[root.value] = root


        for i in range(root.value,self.N,-1):

            self.node_dict[i-1] = Node(i-1)


        for i in range(root.value,self.N-1,-1):
            
            # Aqui podemos visualizar os filhos do nosso noh atual
            aux_df = self.linkage[self.linkage['parent'] == i]

            # Teremos um filho na esquerda e um filho na direita
            right_child = aux_df['right_child'].values[0]
            left_child = aux_df['left_child'].values[0]

            # Isso significa, que o nosso noh corrente, tem dois filhos não individuais
            if right_child >= self.N and left_child >= self.N:
                # ID do filho da esquerda
                self.node_dict[i].left = self.node_dict[left_child]
                self.node_dict[i].right = self.node_dict[right_child]

                # Atribui a condição de PAI
                self.node_dict[left_child].parent = self.node_dict[i]
                self.node_dict[right_child].parent = self.node_dict[i]
            
            # Isso significa, que o nosso noh corrente, tem um filho individual a direita e um não individual a esquerda
            elif right_child < self.N and left_child >= self.N:

                self.node_dict[i].left = self.node_dict[left_child]

                # Atribui a condição de PAI
                self.node_dict[left_child].parent = self.node_dict[i]

            # Isso significa, que o nosso noh corrente, tem um filho individual a esquerda e um não individual a direita
            elif right_child >= self.N and left_child < self.N:

                self.node_dict[i].right = self.node_dict[right_child]

                # Atribui a condição de PAI
                self.node_dict[right_child].parent = self.node_dict[i]

            # Isso significa, que o nosso noh corrente tem dois filhos  individuais
            else:
                continue

        # Inserir as arrays

        current_node = self.node_dict[self.N]

        for i in range(self.N,nodes):
            
            row = self.linkage[self.linkage['parent'] == current_node.value]

            # Teremos um filho na esquerda e um filho na direita
            right_child = row['right_child'].values[0]
            left_child = row['left_child'].values[0]


            # Isso significa, que o nosso noh corrente, tem dois filhos não individuais
            if right_child >= self.N and left_child >= self.N:

                """array_left = current_node.left.indexes
                array_right = current_node.right.indexes


                current_node.indexes = np.concatenate((array_left,array_right))
                current_node.size = len(current_node.indexes)"""

                array_right_size = current_node.right.size 
                array_left_size = current_node.left.size 

                current_node.size = array_right_size + array_left_size
            
            # Isso significa, que o nosso noh corrente, tem um filho individual a direita e um não individual a esquerda
            elif right_child < self.N and left_child >= self.N:

                """array_left = current_node.left.indexes


                # Atribui a condição de PAI
                current_node.indexes = np.concatenate((array_left,[right_child]))
                current_node.size = len(current_node.indexes)"""

                array_left_size = current_node.left.size 

                current_node.size = array_left_size + 1

            # Isso significa, que o nosso noh corrente, tem um filho individual a esquerda e um não individual a direita
            elif right_child >= self.N and left_child < self.N:

                """array_right = current_node.right.indexes


                # Atribui a condição de PAI
                current_node.indexes = np.concatenate((array_right,[left_child]))
                current_node.size = len(current_node.indexes)"""

                array_right_size = current_node.right.size 

                current_node.size = array_right_size + 1

            # Isso significa, que o nosso noh corrente tem dois filhos  individuais
            else:
                
                """current_node.indexes = [left_child,right_child]

                current_node.indexes.sort()"""

                current_node.size = 2

            # Troca o noh

            if current_node.value + 1 != nodes:
                current_node = self.node_dict[current_node.value+1]

        # Hierarquia montada com cada nó contendo os indices.
        return


    

    def get_hierarchy(self,obj_id):

        final_id = self.linkage[ (self.linkage['left_child'] == (obj_id)) | (self.linkage['right_child'] == (obj_id))]['parent'].values[0]

        current_node = self.node_dict[final_id]

        lista = [current_node.value]

        while current_node.parent != None:

            current_node = current_node.parent

            lista.append(current_node.value)

        # Retorna a lista contendo as hierarquias    
        return np.array(lista)

    def extract_hierarchy(self):

        new_node_list = {}

        # Construímos novos nohs.
        for i in range(self.N):

            new_node_list[i] = Node(i)

        # Adicionaremos os nohs que cada objeto está presente no atributo index
        for i in range(self.N):

            new_node_list[i].indexes = self.get_hierarchy(i)

        return new_node_list

class Evaluate():

    def __init__(self) -> None:

        
        pass


    def ARI_val(self,cluster_true, cluster_pred):

        from sklearn.metrics.cluster import adjusted_rand_score


        self.ari = adjusted_rand_score(cluster_true, cluster_pred)

        return self.ari


    
    # Compara duas hierarquias
    def HAI_val(self,HAI1, HAI2):

        if HAI2.N != HAI1.N:
            print("Hierarquias com tamanhos diferentes, impossível realizar o cálculo.")
            return

        self.soma = .0


        self.H1_new_node_dict = HAI1.extract_hierarchy()
        self.H2_new_node_dict = HAI2.extract_hierarchy()


        N = HAI1.N


        t0 = time.time()
        

        for obj_id in range(N):

            exact_path = self.H1_new_node_dict[obj_id].indexes
            app_path = self.H2_new_node_dict[obj_id].indexes

            print(f"\r{obj_id}/{N} ({(time.time()-t0):.3f} s)" , end=' ')


            for j in range(N):
                

                if obj_id < j:

                    j_exact_path = self.H1_new_node_dict[j].indexes
                    j_app_path = self.H2_new_node_dict[j].indexes
                    
                    
                   # exact_inter_id = np.intersect1d(exact_path, j_exact_path)[0]
                    # app_inter_id = np.intersect1d(app_path,j_app_path)[0]


                    exact_inter_id = intersect1d_searchsorted(exact_path, j_exact_path)
                    app_inter_id = intersect1d_searchsorted(app_path,j_app_path)


                    exact_node_size = HAI1.node_dict[exact_inter_id].size
                    app_node_size = HAI2.node_dict[app_inter_id].size

                    # Temos o nível da hierarquia que os pontos se encontram
                    d1 = (exact_node_size) / float(N)

                    d2 = (app_node_size) / float(N)


                    self.soma += 2*(math.fabs(d1-d2))
  
        return (1 - (1.0/(N**2) * self.soma))


