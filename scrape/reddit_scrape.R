# for extracting reddit posts
library(RedditExtractoR)
library(tidyverse)

# ===============================================================================
# Scraping the sub-reddits 
# ===============================================================================

# INITIAL CODE TO SCRAPE THE REDDIT DATA USING REDDITEXTRACTOR, 

climate_change <- find_thread_urls(
  subreddit = "climatechange", sort_by = "new", period = "day")

nba <- find_thread_urls(
  subreddit = 'nba', sort_by = 'new', period = 'day')

guns <- find_thread_urls(
  subreddit = 'guns', sort_by = "new", period = 'day')

# save to each college's posts to one r data file
all_posts <- bind_rows(climate_change, nba, guns)


write.csv(x=all_posts, file = 'all_posts.csv')