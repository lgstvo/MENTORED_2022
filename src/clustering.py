from sklearn.cluster import KMeans, DBSCAN, Birch, AgglomerativeClustering, SpectralClustering
from sklearn_som.som import SOM
import scipy.stats as scipy
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, calinski_harabasz_score, silhouette_score, davies_bouldin_score

class Clustering():
    
    def __init__(self, data_csv, dataset, infected):
        self.dataset = data_csv
        if dataset == "capture52":
            self.__defineGT__(["147.32.84.165", "147.32.84.191", "147.32.84.192"])
            self.dataset = self.process_entropy()
        elif dataset == "capture51":
            self.infected = infected
        elif dataset == "cic":
            self.infected = infected
            #self.__defineGT__(["192.168.50.4"])
            #self.dataset = self.process_entropy()
        self.__compute_PCA__()
        self.__clust_alg = None

    def __feature_select__(self):
        self.dataset = self.dataset[["Source_int", "Destination_int", "Source_Port", "Destination_Port", "Length", "point_id"]]
    
    def __compute_PCA__(self):
        data = self.dataset.to_numpy()
        pca = PCA(n_components = 3)
        data = data.transpose()
        pca.fit(data)
        self.pca_points = pca.components_.transpose()
        print(self.pca_points)
       

    def __defineGT__(self, botnets):
        n_points_total = self.dataset["point_id"].values[-1]
        infected_list = []
        for point in range(n_points_total+1):
            infected = 0
            source_ips = self.dataset[self.dataset["point_id"] == point]["Source"]
            for ip in source_ips:
                for bot in botnets:
                    if ip == bot:
                        infected = 1
            infected_list.append(infected)
        self.infected = infected_list

    def process_entropy(self):
        self.__feature_select__()
        entropy_dframe = pd.DataFrame()
        n_points_total = self.dataset["point_id"].values[-1]
        for point in range(n_points_total+1):
            window = self.dataset[self.dataset["point_id"] == point]
            entropy_features = []
            for (column_name, column_data) in window.iteritems():
                if column_name == "point_id":
                    continue
                column_entropy = scipy.entropy(column_data)
                entropy_features.append(column_entropy)
            dframe_model = {
                "Source_int":       [entropy_features[0]],
                "Destination_int":  [entropy_features[1]],
                #"Source_Port":      [entropy_features[2]],
                #"Destination_Port": [entropy_features[3]],
                "Length":           [entropy_features[4]]
            }
            entropy_dframe_row = pd.DataFrame(dframe_model)
            entropy_dframe = pd.concat([entropy_dframe, entropy_dframe_row], ignore_index=True, axis=0)
        
        print(entropy_dframe)
        return entropy_dframe

    def __load_kmeans__(self, n_clusters=2, random_state=0):
        self.__clust_alg = KMeans(n_clusters=n_clusters, random_state=random_state)

    def __load_dbscan__(self, eps=0.005, min_samples=40):
        self.__clust_alg = DBSCAN(eps=eps, min_samples=min_samples)

    def __load_som__(self, m=1, n=2, dim=2):
        self.__clust_alg = SOM(m=m, n=n, dim=dim)

    def __load_birch__(self, n_clusters=2):
        self.__clust_alg = Birch(n_clusters=n_clusters, threshold=0.01)

    def __load_ward__(self, n_clusters=2):
        self.__clust_alg = AgglomerativeClustering(n_clusters=n_clusters)

    def __load_spectral__(self, n_clusters=2, random_state=0):
        self.__clust_alg = SpectralClustering(n_clusters=n_clusters, random_state=random_state)

    def evaluate(self, pts, label):
        calinski_harabasz = calinski_harabasz_score(pts, label)
        silhouette = silhouette_score(pts, label)
        davies_bouldin = davies_bouldin_score(pts, label)
        
        print("Davies Bouldin Socre: ", davies_bouldin)
        print("Silhouette Score: ", silhouette)
        print("Calinski Harabasz Score: ", calinski_harabasz)

    def clusterize(self, title_str, t=-1):
        if self.__clust_alg == None:
            print("Please define clustering algorithm.")
            return

        if t != -1:
            pts = self.pca_points[t-300:t]
        else:
            pts = self.pca_points
        
        clusters = self.__clust_alg.fit_predict(pts[:,:2])
        
        self.evaluate(pts, clusters)
        plt.scatter(pts[:, 0], pts[:, 1], c=clusters)
        plt.savefig("./img/{}_{}.png".format(title_str, self.method))
        plt.close()

    def ground_truth(self, title_str, t=-1):
        if t != -1:
            pts = self.pca_points[t-300:t]
            infected = self.infected[t-300:t]
        else:
            pts = self.pca_points
            infected = self.infected

        plt.scatter(pts[:, 0], pts[:, 1], c=infected)
        plt.savefig("./img/{}_GT.png".format(title_str))
        plt.close()

    def confusion_m(self, title_str, t=-1):
        if t != -1:
            pts = self.pca_points[t-300:t]
            infected = self.infected[t-300:t]
        else:
            pts = self.pca_points
            infected = self.infected

        predicted = self.__clust_alg.fit_predict(pts[:,:2])

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        itp = 0
        itn = 0
        ifp = 0
        ifn = 0

        for index, value in enumerate(predicted):
            gt = infected[index]

            if gt == 0 and value == 0:
                tn = tn + 1
                ifn = ifn + 1
            elif gt == 0 and value != 0:
                fp = fp + 1
                itp =  itp + 1
            elif gt != 0 and value == 0:
                fn = fn + 1
                itn = itn + 1
            elif gt != 0 and value != 0:
                tp = tp + 1
                ifp = ifp + 1

        cm = [[tp, fp], [fn, tn]]
        icm = [[itp, ifp], [ifn, itn]]

        #tpr = tp/(tp+fn)
        #fpr = fp/(fp+tn)
        #tnr = tn/(tn+fp)
        #fnr = fn/(fn+tp)
        #ppv = tp/(tp+fp)
        #fdr = fp/(fp+tp)
        #print("True Positive Prc: {}\n".format(tp/len(predicted)))
        #print("True Negative Prc: {}\n".format(tn/len(predicted)))
        #print("False Positive Prc: {}\n".format(fp/len(predicted)))
        #print("False Negative Prc: {}\n".format(fn/len(predicted)))
        #print("Recall: {}\n".format(tpr))
        #print("Precision: {}\n".format(ppv))
        #print("False Discovery Rate: {}\n".format(fdr))
        
        print(np.unique(predicted))
        print(np.unique(infected))
        print(confusion_matrix(infected, predicted))
        print(classification_report(infected, predicted))
        
        #print("Positive Likelihodd: {}\n".format(tpr/fpr))
        #print("Negative Likelihodd: {}\n".format(tnr/fnr))

        # print("Inverted True Positive Rate: {}".format(itp/len(predicted)))
        # print("Inverted True Negative Rate: {}".format(itn/len(predicted)))
        # print("Inverted False Positive Rate: {}".format(ifp/len(predicted)))
        # print("Inverted False Negative Rate: {}".format(ifn/len(predicted)))
        '''
        plt.plot(cm)
        plt.xlabel('Predictions')
        plt.ylabel('Actuals')
        plt.title('Confusion Matrix')
        plt.savefig("./img/{}_{}_CM.png".format(title_str, self.method))

        plt.plot(icm)
        plt.xlabel('Predictions')
        plt.ylabel('Actuals')
        plt.title('Inverted Confusion Matrix')
        plt.savefig("./img/{}_{}_ICM.png".format(title_str, self.method))
        plt.close()
    
        '''
        return cm, icm


    def load_method(self, method):
        self.method = method
        if method == "KMeans":
            self.__load_kmeans__()
        elif method == "DBSCAN":
            self.__load_dbscan__()
        elif method == "SOM":
            self.__load_som__()
        elif method == "Birch":
            self.__load_birch__()
        elif method == "Ward":
            self.__load_ward__()
        elif method == "Spectral":
            self.__load_spectral__()