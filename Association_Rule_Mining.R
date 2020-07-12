# Assignment 2: Association Rule Mining
## Association Rule Mining Using R: Online radio music recommendation

# Question 1:
# Load the music data into a sparse matrix
music <- read.transactions(file.choose(), sep = ",", rm.duplicates = TRUE)

# Question 2:
# inspect the sparse matrix
summary(music)
## users are 14593 and columns are 1004
## the density here means that in this matrix, 1.97% of cells are having values and the value is 1.
## the most frequent artist/item is radiohead : 2703. 
## 340 users have 30 items/artists.

# Question 3:  
# Inspect the first 5 users’ artists.
inspect(music[1:5])
## yes, it matches the original csv file.
# check the frequencies/supports of users’ artists in column from 4 to 7
itemFrequency(music[,4:7])
## 3 doors down: 0.031796067
## 30 seconds to mars: 0.033714795
## 311: 0.008497225
## 36 crazyfists: 0.008086069
                         
# Question 4: 
# Create a histogram plotting the artists have more than 10 percent support (itemFrequencyPlot)
itemFrequencyPlot(music, support = 0.1)
## there are 10 artists in the matrix with at least 10% support.
# plot top 20 artists with highest support
itemFrequencyPlot(music, topN = 20)
## Artist with 15th highest support is Placebo: 0.079627219.

# Question 5:
# (image) of the sparse matrix for the first 100 users' preference
image(music[1:100]) 
# a visualization of a random sample of 500 users' artist selection
image(sample(music, 500))

# Question 6:
# apriori() function to train the association rules on the music data
apriori(music)
## zero rules are generated.
# adjust support level to 0.01, minlen =2 and confidence level to 0.25.
musicrules <- apriori(music, parameter = list(support= 0.01, confidence = 0.25, minlen = 2))
musicrules
## 788 rules are generated.


# Question 7: 
# Summarize the rules generated from adjusted parameters
summary(musicrules)
## there are 224 rules with size = 3
# check the first ten rules
inspect(musicrules[1:10])
## if a user likes "the pussycat dolls" then "rihanna" artist should be recommended to this user.
# Sort the rule list by lift and inspect the first five rules with highest lift
inspect(sort(musicrules, by = "lift")[1:5])
## {beyoncc} => {rihanna} has the fourth highest rule.

#  Question 8:
#  Find subsets of rules containing any cold play
firstrules <- subset(musicrules, items %in% "coldplay")
firstrules
## there are 172 rules which includes "coldplay"
# Sort the rules by support and inspect the first five cold play rules with highest support
inspect(sort(firstrules, by = "support")[1:5])
## {radiohead} => {coldplay} has the 2nd highest support.

# Question 9: 
# write these rules to a csv file and save the rules into a data frame
write(musicrules, file = "musicrules.csv",
      sep =",", quote = TRUE, row.names = FALSE)
musicrules_df <- as(musicrules, "data.frame")
str(musicrules_df)
## there are 5 variables in the data frame
