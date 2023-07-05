library(data.table)
library(tidyverse)
library(tidytext)
library(textmineR)
library(tm)
library(topicmodels)
library(dplyr)
library(slam)
library(cluster)
library(gtools)
library(tidyr)
library(aricode)
library(mclust)
library(MCMCpack)
if (!requireNamespace("forcats", quietly = TRUE)) {
  install.packages("forcats")
}

library(forcats)

# Reading the original data sets
data_true <- read.csv("C:/Users/theal/Desktop/olx/gregor/DataSet_Misinfo_TRUE.csv")
data_false <- read.csv("C:/Users/theal/Desktop/olx/gregor/DataSet_Misinfo_FAKE.csv")
data_true = data_true[-1,]
data_false = data_false[-1,]
colnames(data_true) = c("ID", "text")
colnames(data_false) = c("ID", "text")

filtered_data <- data_true[, c("ID", "text")]

# Naming the data
data = rbind(filtered_data, data_false)
label = c(rep("True", nrow(filtered_data)), 
          rep("False", nrow(data_false)))

# Sampling a subset of the data for easier computational times
set.seed(7)
index.sample = sample(1:nrow(data), 5000, replace = FALSE)
data = data[index.sample,]
label = label[index.sample]
label = cbind(data[,1], label)

# Removing rows with same Document ID
i = 1
unique.rows = length(unique(data$ID))
while(i < unique.rows+1){
  ind = which(data$ID == data$ID[i])
  if(length(ind) > 1){
    data = data[-ind[-1],]
    label = label[-ind[-1],]
  }
  i = i+1
}


# Cleaning Data removing stop words and so on

text_cleaning_tokens <- data %>% 
  tidytext::unnest_tokens(word, text)
text_cleaning_tokens$word <- gsub('[[:digit:]]+', '', text_cleaning_tokens$word)
text_cleaning_tokens$word <- gsub('[[:punct:]]+', '', text_cleaning_tokens$word)
text_cleaning_tokens <- text_cleaning_tokens %>% filter(!(nchar(word) == 1))%>% 
  anti_join(stop_words)
tokens <- text_cleaning_tokens %>% filter(!(word==""))
tokens <- tokens %>% mutate(ind = row_number())
tokens <- tokens %>% group_by(ID) %>% mutate(ind = row_number()) %>%
  tidyr::spread(key = ind, value = word)
tokens[is.na(tokens)] <- ""
tokens <- tidyr::unite(tokens, text,-ID,sep =" " )
tokens$text <- trimws(tokens$text)

# create DTM - (document term matrix), 
# which is a sparse matrix containing the terms/words as columns 
# and documents as rows.
dtm <- CreateDtm(tokens$text, 
                 doc_names = tokens$ID, 
                 ngram_window = c(1, 2))

remaining_index = which(data$ID %in% tokens$ID)
MisinformationData = data[remaining_index,]
MisinformationLabel = label[remaining_index,]

# Exploring the basic frequency of words in the documents
tf <- TermDocFreq(dtm = dtm)
original_tf <- data.frame(term = tf$term, term_freq = tf$term_freq, doc_freq = tf$doc_freq)
rownames(original_tf) <- 1:nrow(original_tf)
# Eliminate words appearing less than 2 times or in more than half of the
# documents
vocabulary <- tf$term[ tf$term_freq > 1 & tf$doc_freq < nrow(dtm) / 2 ]
dtm = dtm[, tf$term_freq > 1 & tf$doc_freq < nrow(dtm) / 2]
Misinfo.DTM = dtm

# Saving the data
save(Misinfo.DTM, MisinformationData, MisinformationLabel, 
     file = "Misinformation.Rdata")

#Zadanie 1a
#Tworzenie modelu LDA z 6 tematami, a następnie wyodrębnienie wartości alpha, beta i gamma z modelu.
lda_model <- LDA(Misinfo.DTM, k = 6, control = list(seed = 30))
alpha <- lda_model@alpha
beta <- lda_model@beta
gamma <- lda_model@gamma

#Tworzenie ramki danych zawierającej 10 najbardziej istotnych słów dla każdego z tematów.
library(tidytext)

term_matrix <- terms(lda_model, 10)

term_df <- term_matrix %>%
  as.data.frame() %>%
  mutate(topic = 1:n()) %>%
  gather(term, beta, -topic) %>%
  arrange(topic, desc(beta)) %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic))
#Wygenerowanie wykresu przedstawiającego 10 najbardziej istotnych słów dla każdego tematu.
term_df %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_y_reordered() +
  theme(strip.text.x = element_text(size = 8))

tidy_lda <- tidy(lda_model)

# Wybierz 10 najbardziej istotnych słów dla każdego tematu
top_terms <- tidy_lda %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup()

top_terms_sorted <- top_terms %>%
  mutate(topic = paste("Temat", topic), 
         term = reorder_within(term, +beta, topic))

# Wygenerowanie wykresu przedstawiającego 10 najbardziej istotnych słów dla każdego tematu, posortowane malejąco według prawdopodobieństwa (beta).
ggplot(top_terms_sorted, aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  theme(strip.text.x = element_text(size = 8)) +
  scale_x_reordered()


#Podział danych na zbiory treningowe i testowe.
set.seed(30)
split_idx <- sample(1:nrow(Misinfo.DTM), size = floor(0.3 * nrow(Misinfo.DTM)))
train_DTM <- Misinfo.DTM[split_idx, ]
test_DTM <- Misinfo.DTM[-split_idx, ]
train_simple_triplet <- as.simple_triplet_matrix(train_DTM)
test_simple_triplet <- as.simple_triplet_matrix(test_DTM)

#Zadanie 1b
# Sprawdzenie różnych wartości k (ilość tematów) w modelu LDA i obliczenie perpleksji dla zbiorów treningowego i testowego.
k_values <- c(2, 4, 6, 8, 10)
lda_models <- lapply(k_values, function(k) {
  LDA(train_simple_triplet, k = k, control = list(seed = 30))
})

train_perplexities <- sapply(lda_models, function(model) {
  perplexity(model, train_simple_triplet)
})
test_perplexities <- sapply(lda_models, function(model) {
  perplexity(model, test_simple_triplet)
})


data.frame(K = k_values,
           Train_Perplexity = train_perplexities,
           Test_Perplexity = test_perplexities) %>%
  gather(key = "Dataset", value = "Perplexity", -K) %>%
  ggplot(aes(x = K, y = Perplexity, color = Dataset)) +
  geom_line() +
  geom_point() +
  labs(title = "Perplexity vs Number of Topics",
       x = "Number of Topics",
       y = "Perplexity")

#Zadanie 2a
#Wykorzystanie klasteryzacji k-średnich do podziału danych na klastry. Następnie obliczenie indeksu ARI (Adjusted Rand Index) dla uzyskanych wyników.

lda_gamma <- lda_model@gamma
kmeans_clusters <- kmeans(lda_gamma, centers = 2)
misinfo_labels_updated <- label[remaining_index, 2]

print(dim(lda_gamma))
print(length(remaining_index))
print(length(kmeans_clusters$cluster))
print(length(misinfo_labels_updated))

str(kmeans_clusters)

print(dim(train_DTM))
print(dim(test_DTM))

kmeans_clusters <- kmeans(lda_gamma, centers = 2)
print(length(kmeans_clusters$cluster))



misinfo_labels <- MisinformationData$MisinformationLabel
print(length(kmeans_clusters$cluster))
print(length(misinfo_labels_updated))

kmeans_clusters_vector <- kmeans_clusters$cluster
misinfo_labels_vector <- unlist(misinfo_labels_updated)


print(MisinformationLabel)
adjusted_rand_index <- adjustedRandIndex(kmeans_clusters_vector, misinfo_labels_vector)
print(adjusted_rand_index)


plot(lda_gamma, col = kmeans_clusters$cluster, main = "Klastry LDA")

# Elbow method
wss <- sapply(1:10, function(k) {
  kmeans(lda_gamma, centers = k, nstart = 25)$tot.withinss
})
plot(1:10, wss, type = "b", xlab = "Number of Clusters", ylab = "Total within-clusters sum of squares", main = "Elbow Method")

# Silhouette method
avg_sil <- sapply(2:10, function(k) {
  silhouette_kmeans <- silhouette(kmeans(lda_gamma, centers = k, nstart = 25)$cluster, dist(lda_gamma))
  mean(silhouette_kmeans[, 3])
})
plot(2:10, avg_sil, type = "b", xlab = "Number of Clusters", ylab = "Average Silhouette Width", main = "Silhouette Method")

#Zadanie 2b
#Dodanie szumu do danych i porównanie wyników klasyfikacji z użyciem lasu losowego oraz indeksu ARI.
noise <- matrix(rnorm(nrow(lda_gamma) * 10), nrow = nrow(lda_gamma), ncol = 10)
misinfo_noisy <- cbind(lda_gamma, noise)

library(randomForest)

print(nrow(MisinformationLabel))
print(nrow(misinfo_noisy))

print(colnames(MisinformationLabel))
print(colnames(misinfo_noisy))
str(MisinformationLabel)
str(misinfo_noisy)

colnames(misinfo_noisy) <- paste0("V", 1:16)
colnames(MisinformationLabel)[1] <- "id"
misinfo_noisy_df <- as.data.frame(misinfo_noisy)
combined_data <- cbind(MisinformationLabel, misinfo_noisy_df)

combined_data$label <- as.factor(combined_data$label)
rf_model <- randomForest(label ~ ., data = combined_data, importance = TRUE)


oob_error <- rf_model$err.rate[nrow(rf_model$err.rate), "OOB"]


kmeans_noisy_clusters <- kmeans(misinfo_noisy, centers = 2)
adjusted_rand_index_noisy <- adjustedRandIndex(kmeans_noisy_clusters$cluster, misinfo_labels_updated)

error_diff <- abs(adjusted_rand_index_noisy - (1 - oob_error))
generate_prob_vector <- function(length) {
  vec <- runif(length)
  vec <- vec / sum(vec)
  return(vec)
}
#Zadanie 3a
#Symulowanie danych z modelu LDA.
simuluj_LDA <- function(M, N, lda_model) {
  alpha <- lda_model@alpha
  beta <- lda_model@beta
  
  simulated_data <- matrix(0, nrow = M, ncol = N)
  
  for (i in 1:M) {
    theta <- generate_prob_vector(nrow(beta))
    
    for (j in 1:N) {
      topic <- sample.int(nrow(beta), 1, prob = theta)
      probs <- beta[topic, ]
      probs <- probs / sum(probs)
      
      word <- sample.int(ncol(beta), 1, prob = probs)  # wybierz losowe słowo
      simulated_data[i, j] <- word
    }
  }
  
  # zmień simulated_data na wektor indeksów kolumn
  simulated_data <- as.vector(simulated_data)
  
  # utwórz rzadką macierz z wektora simulated_data
  simulated_dtm <- sparseMatrix(i = rep(1:M, each = N), j = seq_along(simulated_data), x = 1, dims = c(M, ncol(beta)))
  
  return(simulated_dtm)
}

# Symulowanie danych
M <- 10 #nrow(Misinfo.DTM) #zamienione z uwagi na brak mocy kalkulacyjnych sprzętu
N <- 1
print(M)
print(N)
print(lda_model)

simulated_dtm <- symuluj_LDA(M, N, lda_model)

original_topics <- apply(lda_gamma, 1, which.max)

# Dopasuj model LDA do simulated_dtm
simulated_lda_model <- LDA(simulated_dtm, k = 6, control = list(seed = 30))
simulated_gamma <- simulated_lda_model@gamma
simulated_topics <- apply(simulated_gamma, 1, which.max)

#Zadanie 3b
# Wykresy
#Porównanie oryginalnych i symulowanych danych przy użyciu histogramów, wykresów QQ i wykresów punktowych.
par(mfrow = c(2, 2))

hist(original_topics, main = "Original Data", xlab = "Topics")
hist(simulated_topics, main = "Simulated Data", xlab = "Topics")

qqplot(original_topics, simulated_topics, main = "Q-Q Plot", xlab = "Original Topics", ylab = "Simulated Topics")

length(original_topics)
length(simulated_topics)
#Scatter Plot nie działa z powodu ustawienia M=10 w 266 linijce dla symulacji, a w oryginale jest ich ponad 4k+ (długości zmiennych 'x' oraz 'y' różnią się)
plot(original_topics, simulated_topics, main = "Scatter Plot", xlab = "Original Topics", ylab = "Simulated Topics")

#Zadanie 3d
#Wykonanie metody bootstrap dla wartości alpha w modelu LDA i obliczenie 95% przedziałów ufności.
bootstrap_iterations <- 100
bootstrap_alpha <- matrix(0, nrow = bootstrap_iterations, ncol = length(lda_model@alpha))

for (i in 1:bootstrap_iterations) {
  boot_indices <- sample(1:50, size = 50, replace = TRUE)
  boot_dtm <- Misinfo.DTM[boot_indices, ]
  boot_lda <- LDA(boot_dtm, k = 6, method = "Gibbs", control = list(iter = 1000))
  bootstrap_alpha[i, ] <- boot_lda@alpha
}

alpha_CI <- apply(bootstrap_alpha, 2, function(column) quantile(column, c(0.025, 0.975)))
alpha_CI

original_alpha <- lda_model@alpha
in_CI <- sapply(1:length(original_alpha), function(i) original_alpha[i] >= alpha_CI[1, i] & original_alpha[i] <= alpha_CI[2, i])
in_CI



