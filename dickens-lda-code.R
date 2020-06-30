setwd("/Users/alexg/Desktop/Rlesson")

library(SentimentAnalysis)
library(tm) 
library(SnowballC)
library(wordcloud)
library(cluster)
library(rpart)
library(topicmodels)

#read in a tale of two cities txt file

t2c.raw <- readLines("a-tale-of-two-cities.txt")
str(t2c.raw)

#collapse lines into one 
t2c.coll <- paste(readLines("a-tale-of-two-cities.txt"), collapse = " ")
str(t2c.coll)

#Begin cleanup by removing dashes and replacing with a space
t2c.sub.1 <- gsub("-"," ", t2c.coll)
#checking to see what we just did
substr(t2c.sub.1, start = 8000, stop = 9000)

#Lower case all words
t2c.sub.2 <- tolower(t2c.sub.1)
substr(t2c.sub.2, start = 8000, stop = 9000)

stopwords("english")
#Continue cleanup by removing stopwords
t2c.sub.3 <- removeWords(t2c.sub.2, stopwords())
#checking again to see what was changed
substr(t2c.sub.3, start = 8000, stop = 9000)

#Only leave letters numbers and spaces
t2c.sub.4 <- gsub("[^a-zA-Z0-9 ]","",t2c.sub.3)
#checking again to see what was changed
substr(t2c.sub.4, start = 8000, stop = 9000)




#now let's create a stemmed corpus
t2c.corpus.stem <- Corpus(VectorSource(t2c.sub.4))
t2c.corpus.stem <- tm_map(t2c.corpus.stem, stemDocument)
t2c.corpus.stem <- tm_map(t2c.corpus.stem, removeNumbers)
t2c.corpus.stem <- tm_map(t2c.corpus.stem, stripWhitespace)  


#I also want a corpus that is not stemmed to compare results to and see if we end up
#losing info by stemming
t2c.corpus <- Corpus(VectorSource(t2c.sub.4))
t2c.corpus <- tm_map(t2c.corpus, removeNumbers)
t2c.corpus <- tm_map(t2c.corpus, stripWhitespace)  


#create the term document matrix of the stemmed corpus
t2c.tdm.stem <- TermDocumentMatrix(t2c.corpus.stem)

nrow(t2c.tdm.stem)
#there are 6,050 terms in the stemmed term document matrix
inspect(t2c.tdm.stem[1:30,])

#turn the stemmed term document matrix into a matrix
t2c.stem.mat <- as.matrix(t2c.tdm.stem)
t2c.stem.mat.dec <- sort(rowSums(t2c.stem.mat),decreasing=TRUE)
t2c.stem.mat.dat <- data.frame(word = names(t2c.stem.mat.dec),freq=t2c.stem.mat.dec)

#create a word cloud using the stemmed term document matrix's descriptive stats
set.seed(19)
png(filename="stemmed-wordcloud.png", width=900, bg="white")
wordcloud(words = t2c.stem.mat.dat$word, freq = t2c.stem.mat.dat$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
dev.off()

#Visualizing top 10 word frequency in a bar chart
png(filename="word-frequency-stemmed.png", width=900, bg="white")
par(mar=c(5,6,4,1)+.1)
barplot(t2c.stem.mat.dat[1:10,]$freq, las = 1, names.arg = t2c.stem.mat.dat[1:10,]$word,
        col ="lightblue", main ="Most frequent words - Stemmed",
        ylab = "Word frequencies",
        xlab = "Word",
        horiz = FALSE)
dev.off()

#top 10 words by frequency are: said, look, one, hand, lorri, time, defarg, man, will, and upon.
#create the term document matrix of the un-stemmed corpus
t2c.tdm <- TermDocumentMatrix(t2c.corpus)

nrow(t2c.tdm)
#there are 9,609 terms in the un-stemmed corpus.
inspect(t2c.tdm[1:30,])

#let's turn the un-stemmed term document matrix into a matrix
t2c.mat <- as.matrix(t2c.tdm)
t2c.mat.dec <- sort(rowSums(t2c.mat),decreasing=TRUE)
t2c.mat.dat <- data.frame(word = names(t2c.mat.dec),freq=t2c.mat.dec)

#create a word cloud using the un-stemmed term document matrix's descriptive stats
set.seed(19)
wordcloud(words = t2c.mat.dat$word, freq = t2c.mat.dat$freq, min.freq = 10,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

#Visualizing top 10 word frequency in a bar chart
png(filename="word-frequency-unstemmed.png", width=900, bg="white")
par(mar=c(5,6,4,1)+.1)
barplot(t2c.mat.dat[1:10,]$freq, las = 1, names.arg = t2c.mat.dat[1:10,]$word,
        col ="lightblue", main ="Most frequent words - Unstemmed",
        ylab = "Word frequencies",
        xlab = "Word",
        horiz = FALSE)
dev.off()

#Top 10 words by frequency for the un-stemmed corpus are said, one, lorry, will, upon, man, defarge, little, time and now.

#back to classification for the stemmed 
#converting the stemmed document term matrix to just a matrix
t2c.labeled.mat <- as.data.frame(as.matrix(t2c.tdm))


#lets cluster the documents, but first find optimal k
t2c.wss <- numeric(10) 
for (k in 1:10) t2c.wss[k] <- sum(kmeans(t2c.tdm.stem, centers=k)$withinss)
png(filename="wss.png", width=900, bg="white")
par(mar=c(5,6,4,1)+.1)
plot(t2c.wss, type="b")
dev.off()
#3 looks like the optimal value for k




t2c.kmeans <- kmeans(t2c.tdm.stem,3)
t2c.kmeans
t2c.kmeans$cluster 

t2c.tdm.stem$cluster <- t2c.kmeans$cluster
t2c.tdm.stem



#create document term matrix with stemmed corpus
t2c.dtm.stem <- DocumentTermMatrix(t2c.corpus.stem)
ui = unique(t2c.dtm.stem$i)
t2c.dtm.stem.nonempty <- t2c.dtm.stem[ui,]


#Use Gibbs sampling to run LDA using stemmed corpus
set.seed(19)
ldaOut <-LDA(t2c.dtm.stem.nonempty,k=3, method="Gibbs")
t2c.ldaOut.terms <- as.matrix(terms(ldaOut,10))
t2c.ldaOut.terms



#create document term matrix using un-stemmed corpus
t2c.dtm <- DocumentTermMatrix(t2c.corpus)
ui = unique(t2c.dtm$i)
t2c.dtm.nonempty <- t2c.dtm[ui,]


#Use Gibbs sampling to run LDA using un-stemmed corpus
set.seed(19)
ldaOut.ns <-LDA(t2c.dtm.nonempty,k=3, method="Gibbs")
t2c.ldaOut.ns.terms <- as.matrix(terms(ldaOut.ns,10))
t2c.ldaOut.ns.terms
