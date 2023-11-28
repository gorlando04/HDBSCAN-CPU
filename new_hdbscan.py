from dataset import Dataset,Distributions,create_dataset,moons_db,get_20news

from antihub import AntiHub

from app_hdbscan import Approximate_HDBSCAN

from eval import HAI,Evaluate

import numpy as np
import pandas as pd
import networkx as nx
import hdbscan
import sys

import time



# Valor de K
K = 32

D = 2
N = int(1e4)
dataset = 'standart'

args = sys.argv[1:]



while args:
    a = args.pop(0)
    if a == '-name':      dataset = args.pop(0)
    elif a == '-D':       D = int(args.pop(0)) 
    elif  a == '-N':      N = int(args.pop(0)) 
    elif a == '-iter':    _ = args.pop(0)
    elif a == '-mpts':    K = int(args.pop(0)) 
    else:
        print("argument %s unknown" % a, file=sys.stderr)
        sys.exit(1)

# Tratar caso dos datasets reais

if dataset in ['anuran','gas','magic','20news_300','beans'] and (D > 2 or N > 5000):
    exit()



print(f"Argumentos: N = {N} Dim = {D} Dataset_Name = {dataset} e MPTS = {K}")
args = {'N':N,'Dim':D,'name':dataset}


################################################################################################################
#                                                                                                              #
#                                           INSTANTIATE DATASET                                                #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################


def instatiate_dataset(N,D,name):
    

    if name == 'standart': 
        return create_dataset(N,D)
    
   
    elif name == 'moons':
        if D != 2:
            exit()
        return  moons_db(N)  
    
    elif name == 'gaussian-sparse':
        from sklearn.datasets import make_blobs


        # Parameters for the dense clusters
        dense_cluster_1_center = (-80,) * D
        dense_cluster_2_center = (80,) * D
        num_samples_dense = int(N * .3333)

        # Parameters for the sparse cluster
        sparse_cluster_center = (0,) * D
        num_samples_sparse = int(N * .33341)

        # Create data for the dense clusters
        dense_data_1, _ = make_blobs(n_samples=num_samples_dense,n_features=D, centers=[dense_cluster_1_center], cluster_std=0.1)
        dense_data_2, _ = make_blobs(n_samples=num_samples_dense, n_features=D, centers=[dense_cluster_2_center], cluster_std=0.1)

        # Create data for the sparse cluster
        sparse_data, _ = make_blobs(n_samples=num_samples_sparse, n_features=D, centers=[sparse_cluster_center], cluster_std=10.0)

        # Combine the datasets
        final_dataset = np.vstack((dense_data_1, dense_data_2, sparse_data))

        np.random.shuffle(final_dataset)

        return final_dataset
    
    elif name == 'beans':

        # Requirement = pip install ucimlrepo
        from ucimlrepo import fetch_ucirepo 
        
        dry_bean_dataset = fetch_ucirepo(id=602) 

        # data (as pandas dataframes) 
        X = dry_bean_dataset.data.features 

        del dry_bean_dataset

        X = X.to_numpy()

        # Normalizar standart
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X  = scaler.fit_transform(X)

        np.random.shuffle(X)

        return X
    
    elif name == '20news_300':

        return get_20news(d=300)
    
    elif name == 'magic':

        df = pd.read_csv('datasets/magic+gamma+telescope/magic04.data',header=None).loc[:,:9]
        X = df.to_numpy()

        # Normalizar standart
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X  = scaler.fit_transform(X)

        np.random.shuffle(X)

        return X

    elif name == 'gas':

        dfs = []

        sufix = [2011,2012,2013,2014,2015]
        for i in sufix:
            dfs.append(pd.read_csv(f'datasets/gas+turbine+co+and+nox+emission+data+set/gt_{i}.csv',))
        
        df = pd.concat(dfs)

        
        X = df.to_numpy()
        # Normalizar, standart
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X  = scaler.fit_transform(X)

        np.random.shuffle(X)

        return X


    elif name == 'anuran':

        df = pd.read_csv('datasets/archive/Frogs_MFCCs.csv').iloc[:,:-4]

        X = df.to_numpy()

        np.random.shuffle(X)

        return X
    
    dist = {}

    if name == 'beta':
        dist = Distributions.get_beta()

    elif name == 'chi':
        dist = Distributions.get_chi()

    elif name == 'gamma':
        dist = Distributions.get_gamma()

    elif name == 'gumbel':
        dist = Distributions.get_gumbel()


    elif name == 'laplace':
        dist = Distributions.get_laplace()


    elif name == 'logistic':
        dist = Distributions.get_logistic()


    elif name == 'poisson':
        dist = Distributions.get_poison()


    elif name == 'uniform':
        dist = Distributions.get_uniform()


    elif name == 'vonmisses':
        dist = Distributions.get_vonmisses()

    else:
        print("Nenhum dataset encontrado.")
        exit()

    ds = Dataset(dist,N,D,False) 

    ds.create_partial_dataset()

    ds.concatenate_data()

    ds.data = np.unique(ds.get_dataset(),axis=0)
    ds.N = ds.get_dataset().shape[0]

    ds.shuffle_ds()


    return ds.get_dataset()



################################################################################################################
#                                                                                                              #
#                                           EXACT HDBSCAN                                                      #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

class ClusteringHDBSCAN:

    def __init__(self,args):
        self.min_cluster_size = args['min_clust']
        self.exact = args['exact']

        


    def cluster(self,data):
        
        if self.exact:

            self.running_time = time.time()
            self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, algorithm='generic', metric='euclidean' , approx_min_span_tree=False, 
                                             match_reference_implementation=True,cluster_selection_method="eom",
                                allow_single_cluster=False,
                                cluster_selection_epsilon=0.0,
                                max_cluster_size=0)
                                        
    
        self.clusterer.fit(data)

        self.running_time = time.time() - self.running_time
        
        self.labels = self.clusterer.labels_

    def cluster_lelland(self,data):
    

        self.running_time = time.time()
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, algorithm='prims_kdtree', metric='euclidean' , approx_min_span_tree=True, 
                                            match_reference_implementation=False,cluster_selection_method="eom",
                            allow_single_cluster=False,
                            cluster_selection_epsilon=0.0,
                            max_cluster_size=0)
                                        

        self.clusterer.fit(data)

        self.running_time = time.time() - self.running_time
        
        self.labels = self.clusterer.labels_



################################################################################################################
#                                                                                                              #
#                                           ESCOPO PRINCIPAL                                                   #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

def write_df(df,index,info):

    for i in info.keys():
        if i != 'gpu_res' and i != 'data':
            df.loc[index,i] = info[i]

    return


def main(args):

    X = instatiate_dataset(int(args['N']),int(args['Dim']),args['name']).astype(np.float64)
    N = X.shape[0]


    df_gpu = None

    file_name = 'new_comparison_hdbscan_experiments.csv'

    try:   
        df_gpu = pd.read_csv(file_name)
    except:
        print("DF_GPU ainda nao existe, logo vai ser criado")
        df_gpu = pd.DataFrame()

    index_ = df_gpu.shape[0]
    ## Constructing the kNNG

    info = {}


    info['Name'] = args['name']
    info['N_sample'] = X.shape[0]
    info['Dim'] = X.shape[1]


    info['Size (GB)'] =  X.nbytes / 1e9

    info['mpts'] = K

    # Instancia o objeto do tipo AntiHub
    antihub = AntiHub(X)
    antihub.build_kNNG(K+1)

    del antihub

    antihub = AntiHub(X)

    # Construção do kNNG (isso aqui será trocado no futuro)
    
    t0 = time.time()
    print("Vamos construir o kNNG")


    antihub.build_kNNG(K+1)
    tf1 = time.time() - t0
    # Verifica se o kNNG inicial é conexo
    print('\033[1m' + "\nVamos verificar se o novo kNNG eh conexo: " +'\033[0;0m' )

    conexo = antihub.is_connected()

    info['is_initial_kNNG_connected'] = conexo

    if conexo:
        print('\033[32m' + '\033[1m' + "kNNG eh conexo"+'\033[0;0m')

    else:
        print('\033[31m' + '\033[1m' + "kNNG nao eh conexo" + '\033[0;0m')

    # Encontra os índices dos anti-hubs

    t0 = time.time()
    hub_col,_ = antihub.get_hubs()

    # Conecta os antihubs entre si
    antihub.connect_new_nodes(hub_col)

    tf2 = time.time() - t0
    # Verifica se o novo kNNG é conexo
    conexo = antihub.is_connected()

    info['is_new_kNNG_connected'] = conexo


    print("\n")

    if conexo:
        print('\033[32m' + '\033[1m' + "Novo kNNG eh conexo"+'\033[0;0m')

    else:
        write_df(df_gpu,index_,info) 
        df_gpu.to_csv(file_name, index=False)
        print('\033[31m' + '\033[1m' + "Novo kNNG nao eh conexo" + '\033[0;0m')
        exit()

    t0 = time.time()
    # Extrai a MST do grafo usando o algoritmo de PRIM
    MST = nx.minimum_spanning_tree(antihub.grafo,algorithm='prim')

    print("Iniciando a construcao do HDBSCAN aproximado")

    # Calcula o HDBSCAN aproximado
    app_hdbscan = Approximate_HDBSCAN(MST,N,K)
    app_hdbscan.fit()

    tf3 = time.time() - t0

    t_approximate = tf1 + tf2 +tf3
    info['Time_myMethod'] = t_approximate

    print("Finalizando a construcao do HDBSCAN aproximado")


    max_node = 2 * app_hdbscan.linkage.shape[0]
    num_points = max_node - (app_hdbscan.linkage.shape[0] - 1)

    parent_array = np.arange(num_points, max_node + 1)
    app_linkage = pd.DataFrame({'parent': parent_array,
                                'left_child': app_hdbscan.linkage[:,0],
                                'right_child': app_hdbscan.linkage[:,1],
                                'distance': app_hdbscan.linkage[:,2],
                                'size': app_hdbscan.linkage[:,3]})[['parent', 'left_child', 'right_child', 'distance', 'size']]
    
    app_labels = app_hdbscan.labels

    # Libera espaço
    del app_hdbscan,antihub,MST

    # Calcula o HDBSCAN exato
    print("Iniciando a construcao do HDBSCAN exato")
    params = {'min_clust':K,'exact':True}
    exact = ClusteringHDBSCAN(params)
    exact.cluster(X)

    info['Time_exact_HDBSCAN'] = exact.running_time

    print("Finalizando a construcao do HDBSCAN exato")

    # Inicia a construção da hierarquia para o cálculo do HAI
    exact_labels = exact.labels
    exact_linkage = exact.clusterer.single_linkage_tree_.to_pandas()
    


    # Libera espaço
    del exact 

    # Cria objeto para a avaliação dos resultados
    eval = Evaluate()

    # Calcula o ARI
    ari = eval.ARI_val(exact_labels,app_labels)

    
    print(f"Para mpts = {params['min_clust']} ARI = {ari}")

    info['ARI_myMethod'] = ari

    
    Hier_exact = HAI(exact_linkage,N )
    Hier_exact.build_hierarchy()

    Hier_app =  HAI(app_linkage, N)
    Hier_app.build_hierarchy()

    #Calcula o HAI
    hai = eval.HAI_val(Hier_exact,Hier_app)
    print(f"\nPara mpts = {params['min_clust']} HAI = {hai}")

    info['HAI_myMethod'] = hai

    del eval
    del Hier_app
    del app_labels, app_linkage,app_hdbscan

    # Cria objeto para a avaliação dos resultados
    eval = Evaluate()

    # Calcula o HDBSCAN exato
    print("Iniciando a construcao do HDBSCAN Lelland")
    params = {'min_clust':K,'exact':False}
    app = ClusteringHDBSCAN(params)
    app.cluster_lelland(X)

    info['Time_app_HDBSCAN'] = app.running_time

    print("Finalizando a construcao do HDBSCAN aproximado")

    # Inicia a construção da hierarquia para o cálculo do HAI
    app_labels = app.labels
    app_linkage = app.clusterer.single_linkage_tree_.to_pandas()


    # Calcula o ARI
    ari = eval.ARI_val(exact_labels,app_labels)

    
    print(f"Para mpts = {params['min_clust']} ARI = {ari}")

    info['ARI_APP_HDBSCAN'] = ari

    Hier_app =  HAI(app_linkage, N)
    Hier_app.build_hierarchy()

    hai = eval.HAI_val(Hier_exact,Hier_app)
    print(f"\nPara mpts = {params['min_clust']} HAI = {hai}")

    info['HAI_APP_HDBSCAN'] = hai


    write_df(df_gpu,index_,info) 


    df_gpu.to_csv(file_name, index=False)


    return 




if __name__ == '__main__':
    main(args)

    exit()



