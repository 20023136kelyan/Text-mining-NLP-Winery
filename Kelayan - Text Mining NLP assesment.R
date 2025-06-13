
# Project: Text Analytics on Wine Reviews

##############################################################################
########Part 1: Setup - Loading Libraries#####################################
##############################################################################
# This section loads all necessary R packages. The functions used are


# install.packages("tidyverse")
# install.packages("tidytext")
# install.packages("topicmodels")
# install.packages("igraph") # For network graphs
# install.packages("ggraph") # For network graphs

# Load the libraries
library(tidyverse)
library(tidytext)
library(topicmodels)
library(tidyr)
library(igraph) # Load for network analysis
library(ggraph) # Load for network visualization


# Set a seed for reproducibility of random processes like LDA
set.seed(98)


##############################################################################
########Part 2: Data Loading and Preprocessing################################
##############################################################################
# In this part, we load the wine ratings data and prepare it for analysis.
# This includes cleaning the data and creating a price category variable.

# Load the dataset from the provided URL
my_wine_ratings <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-05-28/winemag-data-130k-v2.csv")

# Data Exploration and Cleaning
# Let's get a first look at the data using summary() as seen in class
summary(my_wine_ratings)

# We need to remove rows where 'price' or 'description' are missing,
# as they are essential for our analysis.
my_wine_df <- my_wine_ratings %>%
  select(country, description, points, price, title, variety) %>%
  filter(!is.na(price) & !is.na(description))
print(my_wine_df)

# Create Price Categories
# To compare expensive vs. inexpensive wines, we'll categorize them based on price.
# We'll use quartiles to define the categories.
price_summary <- summary(my_wine_df$price)
first_quartile <- price_summary['1st Qu.']
third_quartile <- price_summary['3rd Qu.']

cat("Price Tiers:\n")
cat("Inexpensive (Bottom 25%): < $", first_quartile, "\n")
cat("Expensive (Top 25%): > $", third_quartile, "\n")

# Create a new column 'price_category'
# We will focus only on the 'Inexpensive' and 'Expensive' categories for a clear comparison.
my_wine_df_filtered <- my_wine_df %>%
  mutate(price_category = case_when(
    price <= first_quartile ~ "Inexpensive",
    price >= third_quartile ~ "Expensive",
    TRUE ~ "Mid-Range" # All other wines fall here
  )) %>%
  filter(price_category %in% c("Inexpensive", "Expensive")) %>%
  # Convert price_category to a factor for proper ordering in plots
  mutate(price_category = factor(price_category, levels = c("Inexpensive", "Expensive")))
print(my_wine_df_filtered)


##############################################################################
########Part 3: Text Mining Framework 1 - Word and Bigram Frequencies#######
##############################################################################
# This framework helps us identify the most commonly used words and phrases
# (bigrams) in the descriptions of expensive and inexpensive wines.

# Tidy the Text Data (Tokenization)

# We break down the descriptions into individual words (tokens).
my_tidy_wine_words <- my_wine_df_filtered %>%
  unnest_tokens(word, description) %>%
  # Remove stop words (common words like "the", "a", "is")
  anti_join(stop_words, by = "word")
print(my_tidy_wine_words)

# Analyze Word Frequencies
word_counts <- my_tidy_wine_words %>%
  count(price_category, word, sort = TRUE)
print(word_counts)

# Visualize Top Words
# Create a bar chart showing the most frequent words for each price category.
# The reorder_within function is from the tidytext package and helps order
# the words correctly within each facet of the plot.
word_counts %>%
  group_by(price_category) %>%
  top_n(15, n) %>%
  ungroup() %>%
  mutate(word = reorder_within(word, n, price_category)) %>%
  ggplot(aes(x = word, y = n, fill = price_category)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~price_category, scales = "free") +
  scale_x_reordered() +
  coord_flip() +
  labs(
    title = "Top 15 Most Frequent Words in Wine Reviews",
    subtitle = "Comparing Inexpensive vs. Expensive Wines",
    x = "Word",
    y = "Frequency"
  ) +
  theme_minimal()

# Analyze Bigram Frequencies

# Bigrams are pairs of consecutive words. This helps find common phrases.
# This approach is similar to the "Episode 6 code starter - N-grams" file.
my_wine_bigrams <- my_wine_df_filtered %>%
  unnest_tokens(bigram, description, token = "ngrams", n = 2)
print(my_wine_bigrams)

# Separate the bigram into two words to remove stop words from each part
my_bigrams_separated <- my_wine_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")
print(my_bigrams_separated)

my_bigrams_filtered <- my_bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)
print(my_bigrams_filtered)

# Count the filtered bigrams
my_bigram_counts <- my_bigrams_filtered %>%
  count(price_category, word1, word2, sort = TRUE) %>%
  # Re-unite the words to form the bigram again
  unite(bigram, word1, word2, sep = " ")
print(my_bigram_counts)

# Visualize Top Bigrams
my_bigram_counts %>%
  group_by(price_category) %>%
  top_n(15, n) %>%
  ungroup() %>%
  mutate(bigram = reorder_within(bigram, n, price_category)) %>%
  ggplot(aes(x = bigram, y = n, fill = price_category)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~price_category, scales = "free_y") +
  scale_x_reordered() +
  coord_flip() +
  labs(
    title = "Top 15 Most Frequent Bigrams in Wine Reviews",
    subtitle = "Common phrases for inexpensive vs. expensive wines",
    x = "Bigram",
    y = "Frequency"
  ) +
  theme_minimal()


##############################################################################
########Part 4: Text Mining Framework 2 - Sentiment Analysis#################
##############################################################################
# This framework examines the emotional tone of the wine descriptions. We use the 'bing' lexicon

# Get the 'bing' sentiment lexicon
bing_sentiments <- get_sentiments("bing")
print(bing_sentiments)

# Perform an inner join to get the sentiment for each word
my_wine_sentiment <- my_tidy_wine_words %>%
  inner_join(bing_sentiments, by = "word")
print(my_wine_sentiment)

# Count positive and negative words for each price category
my_sentiment_counts <- my_wine_sentiment %>%
  count(price_category, sentiment)
print(my_sentiment_counts)

# Visualize Sentiment Comparison
ggplot(my_sentiment_counts, aes(x = sentiment, y = n, fill = price_category)) +
  geom_col(position = "dodge") +
  labs(
    title = "Sentiment Analysis of Wine Descriptions",
    subtitle = "Comparing Inexpensive vs. Expensive Wines using the Bing Lexicon",
    x = "Sentiment",
    y = "Number of Words",
    fill = "Price Category"
  ) +
  theme_minimal()


##############################################################################
########Part 5: Text Mining Framework 3 - Topic Modeling with LDA############
##############################################################################
# Latent Dirichlet Allocation (LDA) is used to find abstract topics in the descriptions. We run it for each
# price category to compare the dominant themes.

# Function to perform and visualize LDA
# Defining a function helps keep the code clean, a practice seen in "Class.R".
udf_perform_and_visualize_lda <- function(data, category_name, num_topics = 4) {
  
  # Create a Document-Term Matrix (DTM)
  dtm <- data %>%
    # Ensure unique document IDs, which is necessary for cast_dtm
    mutate(doc_id = row_number() %>% as.character()) %>% # Convert to character for DTM
    unnest_tokens(word, description) %>%
    anti_join(stop_words, by = "word") %>%
    # Remove words that are too rare or too common (optional, but can improve LDA)
    # For simplicity, we'll skip this for now, but in a real project, consider
    # bind_tf_idf() and filtering by tf_idf or using removeSparseTerms from tm.
    count(doc_id, word, sort = TRUE) %>%
    cast_dtm(doc_id, word, n)
  
  # Run the LDA algorithm
  # Ensure k is not greater than the number of unique terms or documents
  actual_k <- min(num_topics, ncol(dtm) - 1, nrow(dtm) - 1)
  lda_model <- LDA(dtm, k = actual_k, control = list(seed = 123))
  
  # Tidy the LDA output to get the per-topic-per-word probabilities (beta)
  topics <- tidy(lda_model, matrix = "beta")
  print(topics) # Display intermediate result
  
  # Get the top terms for each topic
  top_terms <- topics %>%
    group_by(topic) %>%
    top_n(10, beta) %>%
    ungroup() %>%
    arrange(topic, -beta)
  print(top_terms) # Display intermediate result
  
  # Visualize the top terms in each topic
  plot <- top_terms %>%
    mutate(term = reorder_within(term, beta, topic)) %>%
    ggplot(aes(term, beta, fill = factor(topic))) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~ topic, scales = "free") +
    scale_x_reordered() +
    coord_flip() +
    labs(
      title = paste("LDA Topic Modeling for", category_name, "Wines"),
      subtitle = paste("Top 10 terms for each of the", actual_k, "topics"),
      x = "Term",
      y = "Beta (Probability of term in topic)"
    ) +
    theme_minimal()
  
  print(plot)
  return(plot) # Return the plot object
}

# Run LDA for each category

# Filter data for 'Inexpensive' wines and run the function
my_inexpensive_wines <- my_wine_df_filtered %>% filter(price_category == "Inexpensive")
lda_inexpensive_plot <- udf_perform_and_visualize_lda(my_inexpensive_wines, "Inexpensive")

# Filter data for 'Expensive' wines and run the function
my_expensive_wines <- my_wine_df_filtered %>% filter(price_category == "Expensive")
lda_expensive_plot <- udf_perform_and_visualize_lda(my_expensive_wines, "Expensive")


##############################################################################
########Part 6: Text Mining Framework 4 - Bigram Network Analysis############
##############################################################################
# This framework visualizes the relationships between words, showing how
# often they appear together, providing insights into common conceptual clusters.
# This directly utilizes the `igraph` and `ggraph` packages.

# Re-use my_bigrams_filtered from Part 3

# Filter for bigrams that occur at least a certain number of times to avoid clutter.

min_bigram_n = 20

# For Inexpensive Wines
my_inexpensive_bigram_graph_data <- my_bigrams_filtered %>%
  filter(price_category == "Inexpensive") %>%
  count(word1, word2, sort = TRUE) %>%
  filter(n >= min_bigram_n)
print(my_inexpensive_bigram_graph_data) # Display intermediate result

my_inexpensive_graph <- my_inexpensive_bigram_graph_data %>%
  graph_from_data_frame()

ggraph(my_inexpensive_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, arrow = arrow(type = "closed", length = unit(0.05, "inches")), end_cap = circle(0.05, 'inches'), edge_colour = "gray50") +
  geom_node_point(color = "skyblue", size = 3) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3.5, family = "sans", max.overlaps = 17) +
  labs(title = "Bigram Network for Inexpensive Wines",
       subtitle = paste("Words co-occurring", min_bigram_n, "or more times")) +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))


# For Expensive Wines
my_expensive_bigram_graph_data <- my_bigrams_filtered %>%
  filter(price_category == "Expensive") %>%
  count(word1, word2, sort = TRUE) %>%
  filter(n >= min_bigram_n)
print(my_expensive_bigram_graph_data) # Display intermediate result

my_expensive_graph <- my_expensive_bigram_graph_data %>%
  graph_from_data_frame()

ggraph(my_expensive_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, arrow = arrow(type = "closed", length = unit(0.05, "inches")), end_cap = circle(0.05, 'inches'), edge_colour = "gray50") +
  geom_node_point(color = "lightcoral", size = 3) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3.5, family = "sans", max.overlaps = 30) +
  labs(title = "Bigram Network for Expensive Wines",
       subtitle = paste("Words co-occurring", min_bigram_n, "or more times")) +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

