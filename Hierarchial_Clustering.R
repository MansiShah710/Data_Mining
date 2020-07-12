
## Hierarchial clusterting

# import the data
unemployment <- read.csv(file.choose())
str(unemployment)


##Remove the first column from the original data (because it is a factor)

unemployment_num <- unemployment[,2:3]
unemployment_num <- unemployment[,-1]
unemployment_num <- unemployment[,c(2,3)]
unemployment_num <- unemployment[, c("mean", "stddev")]

## calculate the distance between two objects

d <- dist(unemployment_num)

##  generate the hierarchial clusters

hier_cluster <- hclust(d)

# generate the hier clusters using one command
hier_cluster <- hclust(dist(unemployment[-1]))

## by default, it is complete link: farthest distance as the similarity
# plot the hier_clusters
plot(hier_cluster)

# k = 3: three clusters
k = 3
clusterCut <- cutree(hier_cluster, k)
clusterCut

# Assign the cluster IDs back to the original data 
unemployment$clusterID <- clusterCut
table(unemployment$clusterID)







