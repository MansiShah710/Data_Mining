
# SimpleKMeans using wine data

# Q1.
##Import the data
wine <- read.csv(file.choose())
str(wine)
summary(wine)

# Q2.	
##install RWeka package
install.packages("RWeka")

##Load the package
library(RWeka)

# Set seed to 100. Use SimpleKMeans from RWeka to perform the clustering task for k=3.
# Use the default Euclidean distance function and the initial centroid assignment feature in the SimpleKMeans commands.
k = 3
set.seed(100)
wine_cluster <- SimpleKMeans(wine, Weka_control(N = k, V = TRUE))
wine_cluster

# Q3.	
# Within cluster sum of squared errors: 48.970291155139165
# Based on the Final cluster centroids, there are 60 wines in cluster 0, 55 wines in cluster 1 and 63 wines in cluster 2.
# for attribute Alcohol, mean of cluster 0, cluster 1 and full data is similar but the mean of cluster 2 is low as compared to other clusters.
# for attribute Flavanoids, mean of cluster 0, cluster 2 and full data is similar but the mean of cluster 1 is low.
# for attribute Color, mean of full data and cluster 0 is similar but the mean of cluster 1 is higher and mean of cluster 2 is lower as compared to other clusters.

# Q4.	
##Use Manhanttan distance 
k = 3
wine_Manhanttan <- SimpleKMeans(wine, Weka_control(N=k, V= T, A ="weka.core.ManhattanDistance"))
wine_Manhanttan
# Yes, there is a difference in clustering results. Cluster 0 and cluster 2 have similar number of wines 
# but the cluster 1 has lower number of wines as below:

# cluster 0   cluster 1     cluster 2
# (63.0)       (51.0)       (64.0)

###########################################

# Kmeans using wine data
 
# Q1.
wine <- read.csv(file.choose())
str(wine)
summary(wine)


# Q2.	
# Standardize all variables using scale. 
wine_scale<-lapply(wine, scale)
wine_standard<-as.data.frame(wine_scale)
summary(wine_standard)

# Q3.	
# Set seed to 100. Use kmeans to perform the clustering task for k=3. 
k = 3
set.seed(100)
wine_3means_standard <- kmeans(wine_standard, k)
wine_3means_standard

# Q4.	
# examples belong to cluster 1, 2 and 3 respectively:
wine_3means_standard$size
# examples for cluster 1,2 and 3 as below:
# cluster 1 = 65
# cluster 2 = 62
# cluster 3 = 51

# Q5.
# the total within-cluster-variance and what is the between-cluster-variance:
wine_3means_standard$withinss
# 385.6983 558.6971 326.3537
wine_3means_standard$tot.withinss
# The total within cluster variance is 1270.749
wine_3means_standard$betweenss
# The between cluster variance is 1030.251

#######################################################
# Hierarchical clustering using protein data

# Q1.	
# Import protein.csv. Use structure and summary commands to heck if the file is imported correctly. 
protein <- read.csv(file.choose())
str(protein)
summary(protein)

# Q2.	
# Build the hierarchical clustering using all the attributes except country (use default link values = complete link). 
hier_clusters<-hclust(dist(protein[,-1]))

# Q3.
## Plot the hier_clusters
plot(hier_clusters)
# if the condition is height <= 20 then we could build 4 minimal cluster number.

# Q4.	
## Cut off the tree at 4 cluster numbers using cuttree.
k =  4
clusterCut<-cutree(hier_clusters, k)
clusterCut

# Q5.	Apply the cluster IDs to the original data frame and look at the first 10 records of cluster number and country name.
##Assign the cluster IDs back to the original data
protein$clusterID <- clusterCut
table(protein$clusterID)

# look at the first 10 records of cluster number and country name.
protein[1:10, c("Country", "clusterID")]

# Q6.	Amongst the first ten records, what countries are in the same cluster with “Albania”?
protein[1:10,]
protein[1:10, c("Country", "clusterID")]

# "Albania" is assigned in the cluster 1. Greece and Czechoslovakia share the same cluster with "Albania".

# Q7.
# a data frame with only country and cluster ID in order and order them by cluster ID
orderindex <- order(protein$clusterID)
protein_country_inorder <- data.frame(Country = protein$Country[orderindex],
                                    clusterID = protein$clusterID[orderindex])
protein_country_inorder
# The cluster ID for "E Germany" is 4
# Two countries are in the same cluster with "E Germany". 
# The countries that have the similar protein intakes as “E Germany” are Spain and Portugal.

# Q8.	
## Use average distance as the similarity calculation
cluster_avg <- hclust(dist(protein[,-1]), method = "average")

# Q9.	
# Plot the dendrogram of the new hierarchical clusters. 
plot(cluster_avg)
# 2 is the minimal cluster number we could build if height <= 20.
  
  